package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"enhanced-research-analyzer/internal/clients"
	"enhanced-research-analyzer/internal/config"
	"enhanced-research-analyzer/internal/models"
)

// DSPy-Go inspired structures
type DSPySignature struct {
	InputFields  []string
	OutputFields []string
	Description  string
}

type DSPyModule interface {
	Forward(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error)
	GetSignature() DSPySignature
	Optimize(examples []models.Example) error
}

// DSPy-Go Module: Query Enhancement
type QueryEnhancementModule struct {
	signature DSPySignature
	analyzer  *ResearchAnalyzer
	examples  []models.Example
}

func NewQueryEnhancementModule(analyzer *ResearchAnalyzer) *QueryEnhancementModule {
	return &QueryEnhancementModule{
		signature: DSPySignature{
			InputFields:  []string{"original_query", "domain_context"},
			OutputFields: []string{"expanded_queries", "key_concepts", "search_strategy"},
			Description:  "Enhance research queries for better semantic search results",
		},
		analyzer: analyzer,
	}
}

func (qem *QueryEnhancementModule) GetSignature() DSPySignature {
	return qem.signature
}

func (qem *QueryEnhancementModule) Forward(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	originalQuery := inputs["original_query"].(string)
	domainContext := ""
	if dc, ok := inputs["domain_context"]; ok {
		domainContext = dc.(string)
	}

	prompt := qem.buildEnhancementPrompt(originalQuery, domainContext)

	response, err := qem.analyzer.generateWithGemini(ctx, prompt, 0.3, 1024)
	if err != nil {
		return nil, err
	}

	// Parse the response to extract enhanced queries and concepts
	enhanced := qem.parseEnhancementResponse(response)
	// Fallback expansions for finance domain if model returns no JSON or empty expansions
	if len(enhanced.Expanded) == 0 {
		enhanced.Expanded = qem.financeFallbackQueries(originalQuery)
		if len(enhanced.Concepts) == 0 {
			enhanced.Concepts = []string{"momentum", "trend following", "cross-sectional momentum", "time-series momentum", "risk management"}
		}
	}

	return map[string]interface{}{
		"expanded_queries": enhanced.Expanded,
		"key_concepts":     enhanced.Concepts,
		"original":         originalQuery,
	}, nil
}

func (qem *QueryEnhancementModule) buildEnhancementPrompt(originalQuery, domainContext string) string {
	basePrompt := fmt.Sprintf(`You are a research query enhancement expert. Your task is to expand and improve research queries for better semantic search results.

Original Query: "%s"
Domain Context: %s

Please provide:
1. 3-5 expanded query variations that capture the same semantic meaning but use different terminology
2. Key concepts and synonyms that should be included in the search
3. Related terms that might appear in relevant papers

Format your response as JSON:
{
  "expanded_queries": ["query1", "query2", "query3"],
  "key_concepts": ["concept1", "concept2", "concept3"],
  "related_terms": ["term1", "term2", "term3"]
}

Focus on medical/scientific terminology, synonyms, and related concepts that would appear in research papers.`, originalQuery, domainContext)

	// Add few-shot examples for better performance
	for _, example := range qem.examples {
		if len(qem.examples) > 0 {
			basePrompt += fmt.Sprintf(`

Example:
Input: "%s"
Output: %s`,
				example.Inputs["original_query"],
				example.Outputs["response"])
		}
	}

	return basePrompt
}

func (qem *QueryEnhancementModule) parseEnhancementResponse(response string) models.EnhancedQuery {
	var result struct {
		ExpandedQueries []string `json:"expanded_queries"`
		KeyConcepts     []string `json:"key_concepts"`
		RelatedTerms    []string `json:"related_terms"`
	}

	// Try to parse JSON response
	if err := json.Unmarshal([]byte(response), &result); err != nil {
		// Fallback: simple parsing
		lines := strings.Split(response, "\n")
		enhanced := models.EnhancedQuery{
			Expanded: []string{},
			Concepts: []string{},
		}

		for _, line := range lines {
			line = strings.TrimSpace(line)
			if strings.Contains(strings.ToLower(line), "query") && len(line) > 10 {
				enhanced.Expanded = append(enhanced.Expanded, line)
			}
			if strings.Contains(strings.ToLower(line), "concept") && len(line) > 5 {
				enhanced.Concepts = append(enhanced.Concepts, line)
			}
		}
		return enhanced
	}

	return models.EnhancedQuery{
		Expanded: result.ExpandedQueries,
		Concepts: append(result.KeyConcepts, result.RelatedTerms...),
	}
}

func (qem *QueryEnhancementModule) Optimize(examples []models.Example) error {
	qem.examples = examples
	return nil
}

// DSPy-Go Module: Analysis Enhancement
type AnalysisModule struct {
	signature DSPySignature
	analyzer  *ResearchAnalyzer
	examples  []models.Example
}

func NewAnalysisModule(analyzer *ResearchAnalyzer) *AnalysisModule {
	return &AnalysisModule{
		signature: DSPySignature{
			InputFields:  []string{"article", "original_query", "analysis_focus"},
			OutputFields: []string{"structured_analysis", "key_insights", "relevance_score"},
			Description:  "Provide comprehensive structured analysis of research articles",
		},
		analyzer: analyzer,
	}
}

func (am *AnalysisModule) GetSignature() DSPySignature {
	return am.signature
}

func (am *AnalysisModule) Forward(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	article := inputs["article"].(models.PubMedArticle)
	originalQuery := inputs["original_query"].(string)
	analysisFocus := ""
	if af, ok := inputs["analysis_focus"]; ok {
		analysisFocus = af.(string)
	}

	prompt := am.buildAnalysisPrompt(article, originalQuery, analysisFocus)

	response, err := am.analyzer.generateWithGemini(ctx, prompt, 0.7, 2048)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"structured_analysis": response,
		"article_title":       article.Title,
	}, nil
}

func (am *AnalysisModule) buildAnalysisPrompt(article models.PubMedArticle, originalQuery, focus string) string {
	prompt := fmt.Sprintf(`You are an expert research analyst. Provide a comprehensive, structured analysis of this research article.

ORIGINAL QUERY: "%s"
ANALYSIS FOCUS: %s

ARTICLE DETAILS:
Title: %s
Authors: %s
Journal: %s (%s)
DOI: %s

Abstract: %s

Please provide a detailed analysis with the following structure:

## 🎯 RELEVANCE TO QUERY
[How this article directly addresses the original query]

## 📊 METHODOLOGY & APPROACH  
[Research design, sample size, methods used]

## 🔍 KEY FINDINGS
[Main results and discoveries - be specific with numbers/statistics when available]

## 💡 INSIGHTS & IMPLICATIONS
[What this means for the field, practical applications]

## ⚖️ STRENGTHS & LIMITATIONS
[Study design strengths and potential weaknesses]

## 🔮 FUTURE RESEARCH DIRECTIONS
[What questions remain, suggested follow-up studies]

## 🏆 SIGNIFICANCE RATING
[Rate 1-10 and justify the importance of this work]

## 📚 RELATED WORK RECOMMENDATIONS
[Suggest 2-3 related research areas or papers to explore]

Make your analysis detailed, critical, and accessible. Focus especially on aspects most relevant to: "%s"`,
		originalQuery, focus, article.Title, formatAuthors(article.Authors),
		article.Journal, article.PubDate, article.DOI, article.Abstract, originalQuery)

	return prompt
}

func (am *AnalysisModule) Optimize(examples []models.Example) error {
	am.examples = examples
	return nil
}

// DSPy-Go Module: Follow-up Question Answering
type FollowupQuestionModule struct {
	signature DSPySignature
	analyzer  *ResearchAnalyzer
	examples  []models.Example
}

func NewFollowupQuestionModule(analyzer *ResearchAnalyzer) *FollowupQuestionModule {
	return &FollowupQuestionModule{
		signature: DSPySignature{
			InputFields:  []string{"article", "question", "context"},
			OutputFields: []string{"answer", "confidence", "source_references"},
			Description:  "Answer follow-up questions about research articles",
		},
		analyzer: analyzer,
	}
}

func (fqm *FollowupQuestionModule) GetSignature() DSPySignature {
	return fqm.signature
}

func (fqm *FollowupQuestionModule) Forward(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	article := inputs["article"].(models.PubMedArticle)
	question := inputs["question"].(string)
	context := ""
	if ctxVal, ok := inputs["context"]; ok {
		context = ctxVal.(string)
	}

	prompt := fqm.buildFollowupPrompt(article, question, context)

	response, err := fqm.analyzer.generateWithModelFallback(ctx, prompt, 0.1, 1024)
	if err != nil {
		return nil, err
	}

	// Parse response for structured output
	answer := fqm.parseFollowupResponse(response)

	return map[string]interface{}{
		"answer":            answer,
		"confidence":        "high", // Could be enhanced with actual confidence scoring
		"source_references": fmt.Sprintf("Based on: %s", article.Title),
	}, nil
}

func (fqm *FollowupQuestionModule) buildFollowupPrompt(article models.PubMedArticle, question, context string) string {
	prompt := fmt.Sprintf(`You are an expert research assistant specializing in academic and scientific content. Answer the user's question based on the provided research article.

ARTICLE INFORMATION:
Title: %s
Authors: %s
Journal: %s
Publication Date: %s
Abstract: %s

%s
QUESTION: %s

INSTRUCTIONS:
1. FIRST, try to answer based on the article content provided - use direct quotes and reference specific findings when possible (no length limit for article-specific content)
2. If the question cannot be fully answered from the article, provide a general explanation of the concept/topic (200-300 words maximum)
3. Clearly indicate when you're providing general information vs. article-specific content
4. Be concise but comprehensive in your explanations
5. Maintain academic tone and accuracy
6. For general explanations, focus on key concepts, definitions, and practical implications

Answer:`,
		article.Title,
		formatAuthors(article.Authors),
		article.Journal,
		article.PubDate,
		article.Abstract,
		context,
		question)

	return prompt
}

func (fqm *FollowupQuestionModule) parseFollowupResponse(response string) string {
	// For now, return the response as-is
	// Could be enhanced to parse structured responses
	return strings.TrimSpace(response)
}

func (fqm *FollowupQuestionModule) Optimize(examples []models.Example) error {
	fqm.examples = examples
	return nil
}

// ResearchAnalyzer handles the core business logic for research analysis
type ResearchAnalyzer struct {
	ApiKey           string
	HttpClient       *http.Client
	QueryEnhancer    *QueryEnhancementModule
	AnalysisModule   *AnalysisModule
	FollowupModule   *FollowupQuestionModule
	EmbeddingCache   map[string][]float64
	OptimizationData []models.Example
	Verbose          bool
}

func NewResearchAnalyzer(apiKey string) *ResearchAnalyzer {
	analyzer := &ResearchAnalyzer{
		ApiKey: apiKey,
		HttpClient: &http.Client{
			Timeout: 45 * time.Second,
		},
		EmbeddingCache: make(map[string][]float64),
	}
	// Set verbosity from environment
	v := strings.ToLower(strings.TrimSpace(os.Getenv("VERBOSE")))
	if v == "1" || v == "true" || strings.ToLower(strings.TrimSpace(os.Getenv("DEBUG"))) == "1" || strings.ToLower(strings.TrimSpace(os.Getenv("DEBUG"))) == "true" {
		analyzer.Verbose = true
	}

	analyzer.QueryEnhancer = NewQueryEnhancementModule(analyzer)
	analyzer.AnalysisModule = NewAnalysisModule(analyzer)
	analyzer.FollowupModule = NewFollowupQuestionModule(analyzer)

	return analyzer
}

// Enhanced search with query expansion
func (ra *ResearchAnalyzer) EnhancedSearch(ctx context.Context, originalQuery string) ([]models.PubMedArticle, *models.EnhancedQuery, error) {
	fmt.Printf("🔍 Enhancing query: %s\n", originalQuery)

	// Step 1: Enhance the query using DSPy module
	enhancementResult, err := ra.QueryEnhancer.Forward(ctx, map[string]interface{}{
		"original_query": originalQuery,
		"domain_context": "biomedical research",
	})
	if err != nil {
		log.Printf("Query enhancement failed, using original: %v", err)
		enhancementResult = map[string]interface{}{
			"expanded_queries": []string{originalQuery},
			"key_concepts":     []string{},
		}
	}

	expandedQueries := enhancementResult["expanded_queries"].([]string)
	concepts := enhancementResult["key_concepts"].([]string)

	enhancedQuery := &models.EnhancedQuery{
		Original: originalQuery,
		Expanded: expandedQueries,
		Concepts: concepts,
	}

	fmt.Printf("✅ Enhanced query with %d variations and %d key concepts\n",
		len(expandedQueries), len(concepts))

	// Step 2: Search PubMed with multiple query variations
	allArticles := make(map[string]models.PubMedArticle) // Use map to deduplicate

	// Search with original query across sources
	articles, err := ra.searchPapers(ctx, originalQuery)
	if err != nil {
		return nil, enhancedQuery, err
	}
	for _, article := range articles {
		allArticles[article.UID] = article
	}

	// Search with expanded queries
	for i, expandedQuery := range expandedQueries[:min(3, len(expandedQueries))] {
		fmt.Printf("🔍 Searching variation %d: %s\n", i+1, expandedQuery)
		moreArticles, err := ra.searchPapers(ctx, expandedQuery)
		if err != nil {
			log.Printf("Warning: expanded search failed for '%s': %v", expandedQuery, err)
			continue
		}
		for _, article := range moreArticles {
			allArticles[article.UID] = article
		}
		time.Sleep(200 * time.Millisecond) // Rate limiting
	}

	// Convert back to slice
	var finalArticles []models.PubMedArticle
	for _, article := range allArticles {
		finalArticles = append(finalArticles, article)
	}

	fmt.Printf("✅ Found %d unique articles across all queries\n", len(finalArticles))
	return finalArticles, enhancedQuery, nil
}

// Aggregate search across sources with basic deduplication (now concurrent)
func (ra *ResearchAnalyzer) searchPapers(ctx context.Context, query string) ([]models.PubMedArticle, error) {
	var all []models.PubMedArticle

	// Create a cancellable context with timeout to bound total wait
	ctx, cancel := context.WithTimeout(ctx, 6*time.Second)
	defer cancel()

	// Run sources concurrently
	type res struct {
		items []models.PubMedArticle
		name  string
		err   error
	}
	ch := make(chan res, 4)
	var wg sync.WaitGroup

	launch := func(name string, fn func(context.Context, string) ([]models.PubMedArticle, error)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			items, err := fn(ctx, query)
			ch <- res{items: items, name: name, err: err}
		}()
	}

	// Launch searches with rate limiting to avoid API limits
	launch("arXiv", ra.searchArxiv)
	time.Sleep(200 * time.Millisecond) // Rate limiting between launches
	launch("Crossref", ra.searchCrossref)
	time.Sleep(200 * time.Millisecond)
	launch("Semantic Scholar", ra.searchSemanticScholar)
	time.Sleep(200 * time.Millisecond)
	launch("CORE", ra.searchCoreAPI)

	go func() {
		wg.Wait()
		close(ch)
	}()

	for r := range ch {
		if r.err != nil {
			log.Printf("[warn] %s search failed: %v", r.name, r.err)
			// Continue with other sources instead of failing completely
			continue
		}
		if ra.Verbose {
			log.Printf("[debug] %s results: %d", r.name, len(r.items))
		}
		all = append(all, r.items...)
	}

	// Log summary of successful sources
	if ra.Verbose {
		log.Printf("[info] Search completed - total results from all sources: %d", len(all))
	}

	// Deduplicate by UID
	m := make(map[string]models.PubMedArticle)
	for _, a := range all {
		if a.UID == "" {
			continue
		}
		if _, ok := m[a.UID]; !ok {
			m[a.UID] = a
		}
	}
	var dedup []models.PubMedArticle
	for _, a := range m {
		dedup = append(dedup, a)
	}
	if ra.Verbose {
		log.Printf("[debug] Aggregated total after dedup: %d", len(dedup))
	}
	return dedup, nil
}

// Client method wrappers - these will call the clients package functions
func (ra *ResearchAnalyzer) searchArxiv(ctx context.Context, query string) ([]models.PubMedArticle, error) {
	return clients.SearchArxiv(ctx, query, ra.HttpClient)
}

func (ra *ResearchAnalyzer) searchCrossref(ctx context.Context, query string) ([]models.PubMedArticle, error) {
	return clients.SearchCrossref(ctx, query, ra.HttpClient)
}

func (ra *ResearchAnalyzer) searchSemanticScholar(ctx context.Context, query string) ([]models.PubMedArticle, error) {
	return clients.SearchSemanticScholar(ctx, query, ra.HttpClient)
}

func (ra *ResearchAnalyzer) searchCoreAPI(ctx context.Context, query string) ([]models.PubMedArticle, error) {
	return clients.SearchCoreAPI(ctx, query, ra.HttpClient)
}

// General generation method with fallback support
func (ra *ResearchAnalyzer) generateWithGemini(ctx context.Context, prompt string, temperature float64, maxTokens int) (string, error) {
	return ra.generateWithFallback(ctx, prompt, temperature, maxTokens, config.GenerationModel)
}

// generateWithFallback tries primary model first, then backup model if available
func (ra *ResearchAnalyzer) generateWithFallback(ctx context.Context, prompt string, temperature float64, maxTokens int, model string) (string, error) {
	reqBody := models.GeminiGenerateRequest{}
	reqBody.Contents = []struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	}{{
		Parts: []struct {
			Text string `json:"text"`
		}{{Text: prompt}},
	}}

	reqBody.GenerationConfig.Temperature = temperature
	reqBody.GenerationConfig.TopK = 40
	reqBody.GenerationConfig.TopP = 0.95
	reqBody.GenerationConfig.MaxOutputTokens = maxTokens

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/%s:generateContent?key=%s", config.GeminiAPIBase, model, ra.ApiKey)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := ra.HttpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
	}

	var genResp models.GeminiGenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if len(genResp.Candidates) == 0 || len(genResp.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("no content generated")
	}

	return genResp.Candidates[0].Content.Parts[0].Text, nil
}

// generateWithModelFallback tries primary model, then backup if primary fails
func (ra *ResearchAnalyzer) generateWithModelFallback(ctx context.Context, prompt string, temperature float64, maxTokens int) (string, error) {
	backupModel := config.GetBackupModel()
	// Try primary model first
	result, err := ra.generateWithFallback(ctx, prompt, temperature, maxTokens, config.GenerationModel)
	if err != nil {
		log.Printf("Primary model (%s) failed: %v, trying backup model (%s)", config.GenerationModel, err, backupModel)
		// Try backup model
		result, err = ra.generateWithFallback(ctx, prompt, temperature, maxTokens, backupModel)
		if err != nil {
			return "", fmt.Errorf("both primary and backup models failed: %w", err)
		}
		log.Printf("Successfully used backup model (%s)", backupModel)
	}
	return result, nil
}

// financeFallbackQueries generates robust finance-domain expansions when the LLM fails to return JSON
func (qem *QueryEnhancementModule) financeFallbackQueries(q string) []string {
	ql := strings.ToLower(strings.TrimSpace(q))
	var ex []string
	add := func(s string) {
		if strings.TrimSpace(s) != "" {
			ex = append(ex, s)
		}
	}

	containsAny := func(s string, subs ...string) bool {
		for _, sub := range subs {
			if strings.Contains(s, strings.ToLower(sub)) {
				return true
			}
		}
		return false
	}

	if containsAny(ql, "momentum", "trend", "relative strength", "tsmom") {
		add("cross-sectional momentum equities")
		add("time-series momentum futures CTA")
		add("trend following breakout strategy")
		add("volatility-scaled momentum strategy")
		add("momentum factor fama french")
		add("relative strength rotation strategy")
		add("moving average crossover trend following")
		add("momentum with transaction costs slippage")
		add("intraday momentum strategy")
		add("momentum in fx and commodities")
		add("momentum crashes and risk management")
		add("ensemble momentum combining signals")
	} else if containsAny(ql, "mean reversion", "pairs", "stat arb", "statistical arbitrage") {
		add("pairs trading cointegration strategy")
		add("statistical arbitrage equity pairs")
		add("mean reversion z-score bollinger bands")
		add("overnight reversal strategy")
		add("intraday mean reversion microstructure shocks")
		add("cross-asset mean reversion strategy")
		add("market making inventory risk model")
	} else if containsAny(ql, "volatility", "options", "variance") {
		add("volatility targeting portfolio")
		add("garch time varying volatility models")
		add("implied vs realized volatility trading")
		add("variance risk premium strategy")
		add("vix term structure contango backwardation")
		add("delta hedging pnl attribution")
	} else {
		// General quant/trading expansions
		add("factor investing value size momentum quality")
		add("risk parity portfolio optimization")
		add("market microstructure price impact trading costs")
		add("execution algorithms twap vwap implementation shortfall")
		add("alpha combination model stacking ensemble")
		add("regime detection hmm markov switching trading")
		add("transaction costs modeling microstructure")
		add("backtesting walk forward cross validation")
		add("ml for trading gradient boosting xgboost")
		add("nlp news sentiment event driven trading")
	}

	// Always include a cleaned version of the original query
	add(strings.TrimSpace(q))

	// Deduplicate and cap
	seen := make(map[string]bool)
	var out []string
	for _, s := range ex {
		key := strings.ToLower(strings.TrimSpace(s))
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, s)
		if len(out) >= 12 {
			break
		}
	}
	return out
}

func formatAuthors(authors []models.Author) string {
	var names []string
	for _, author := range authors {
		names = append(names, author.Name)
	}
	if len(names) > 3 {
		return strings.Join(names[:3], ", ") + ", et al."
	}
	return strings.Join(names, ", ")
}

// FindRelevantArticles performs advanced similarity analysis and ranking
func (ra *ResearchAnalyzer) FindRelevantArticles(ctx context.Context, enhancedQuery *models.EnhancedQuery, articles []models.PubMedArticle) ([]models.ArticleWithScore, error) {
	fmt.Printf("🧠 Generating advanced embeddings and calculating relevance scores...\n")

	// Generate query embedding with specific task type
	queryEmbedding, err := ra.getEmbedding(ctx, enhancedQuery.Original, "RETRIEVAL_QUERY")
	if err != nil {
		return nil, fmt.Errorf("failed to get query embedding: %w", err)
	}

	var articlesWithScores []models.ArticleWithScore
	var allScored []models.ArticleWithScore
	minSim, maxSim := 1.0, -1.0

	// Process articles with progress indicator
	for i, article := range articles {
		fmt.Printf("Processing article %d/%d...\r", i+1, len(articles))

		// Create rich text for embedding (title + abstract + journal + year)
		year := article.PubDate
		if len(year) >= 4 {
			year = year[:4]
		}
		text := fmt.Sprintf("Title: %s\nAbstract: %s\nVenue: %s\nYear: %s",
			article.Title, article.Abstract, article.Journal, year)

		// Get article embedding with document-specific task type
		articleEmbedding, err := ra.getEmbedding(ctx, text, "RETRIEVAL_DOCUMENT")
		if err != nil {
			log.Printf("Warning: failed to get embedding for article %s: %v", article.Title, err)
			continue
		}

		// Calculate advanced similarity score
		articleWithScore := ra.calculateAdvancedSimilarity(
			queryEmbedding, articleEmbedding, enhancedQuery.Original, article)
		allScored = append(allScored, articleWithScore)
		if articleWithScore.SemanticSimilarity < minSim {
			minSim = articleWithScore.SemanticSimilarity
		}
		if articleWithScore.SemanticSimilarity > maxSim {
			maxSim = articleWithScore.SemanticSimilarity
		}

		// Only include articles above similarity threshold
		if articleWithScore.Score >= 0.35 { // Using config.SimilarityThreshold
			articlesWithScores = append(articlesWithScores, articleWithScore)
		}

		// Rate limiting
		time.Sleep(80 * time.Millisecond)
	}

	// If nothing met the threshold, fall back to top-N by score
	if len(articlesWithScores) == 0 && len(allScored) > 0 {
		if ra.Verbose {
			log.Printf("[debug] No items above threshold %.2f; falling back to top %d by score (sim range %.3f..%.3f)", 0.35, 8, minSim, maxSim)
		}
		sort.Slice(allScored, func(i, j int) bool { return allScored[i].Score > allScored[j].Score })
		limit := 8 // config.TopArticles
		if len(allScored) < limit {
			limit = len(allScored)
		}
		articlesWithScores = allScored[:limit]
	}

	// Sort by combined score (descending)
	sort.Slice(articlesWithScores, func(i, j int) bool {
		return articlesWithScores[i].Score > articlesWithScores[j].Score
	})

	// Return top articles
	if len(articlesWithScores) > 8 {
		articlesWithScores = articlesWithScores[:8]
	}

	fmt.Printf("\n✅ Found %d highly relevant articles (threshold %.2f; sim range %.3f..%.3f)\n\n",
		len(articlesWithScores), 0.35, minSim, maxSim)
	return articlesWithScores, nil
}

// AnalyzeArticle performs comprehensive analysis of a specific article
func (ra *ResearchAnalyzer) AnalyzeArticle(ctx context.Context, article models.PubMedArticle, originalQuery string) (string, error) {
	fmt.Printf("📊 Analyzing article with enhanced DSPy-powered analysis...\n")

	result, err := ra.AnalysisModule.Forward(ctx, map[string]interface{}{
		"article":        article,
		"original_query": originalQuery,
		"analysis_focus": "comprehensive research analysis",
	})
	if err != nil {
		return "", err
	}

	return result["structured_analysis"].(string), nil
}

// AnswerFollowupQuestion answers follow-up questions about a specific article
func (ra *ResearchAnalyzer) AnswerFollowupQuestion(ctx context.Context, article models.PubMedArticle, question string) (string, error) {
	fmt.Printf("🤔 Answering follow-up question using DSPy-powered analysis...\n")

	result, err := ra.FollowupModule.Forward(ctx, map[string]interface{}{
		"article":  article,
		"question": question,
		"context":  "Follow-up question about research article content - provide general explanation if not covered in article",
	})
	if err != nil {
		return "", err
	}

	return result["answer"].(string), nil
}

// AnswerFollowupQuestionWithModel answers follow-up questions using a specific model
func (ra *ResearchAnalyzer) AnswerFollowupQuestionWithModel(ctx context.Context, article models.PubMedArticle, question string, model string) (string, error) {
	fmt.Printf("🤔 Answering follow-up question using %s model...\n", model)

	prompt := ra.FollowupModule.buildFollowupPrompt(article, question, "Follow-up question about research article content - provide general explanation if not covered in article")

	response, err := ra.generateWithFallback(ctx, prompt, 0.1, 1024, model)
	if err != nil {
		return "", err
	}

	return ra.FollowupModule.parseFollowupResponse(response), nil
}

// Helper methods for embeddings and similarity calculation
func (ra *ResearchAnalyzer) getEmbedding(ctx context.Context, text string, taskType string) ([]float64, error) {
	// Check cache first
	cacheKey := fmt.Sprintf("%s:%s", taskType, text)
	if embedding, exists := ra.EmbeddingCache[cacheKey]; exists {
		return embedding, nil
	}

	// Use local embeddings via Ollama
	baseURL := os.Getenv("OLLAMA_BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	apiURL := strings.TrimRight(baseURL, "/") + "/api/embeddings"

	// Build request for Ollama embeddings API
	reqBody := struct {
		Model string `json:"model"`
		Input string `json:"input"`
	}{
		Model: "embeddinggemma:300m", // config.EmbeddingModel
		Input: text,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := ra.HttpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding from Ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Ollama API error %d: %s", resp.StatusCode, string(body))
	}

	var embResp struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
		return nil, fmt.Errorf("failed to decode Ollama embedding response: %w", err)
	}

	// Cache the embedding
	ra.EmbeddingCache[cacheKey] = embResp.Embedding
	return embResp.Embedding, nil
}

func (ra *ResearchAnalyzer) calculateAdvancedSimilarity(queryEmbedding, articleEmbedding []float64,
	originalQuery string, article models.PubMedArticle) models.ArticleWithScore {

	// Cosine similarity
	cosineSim := cosineSimilarity(queryEmbedding, articleEmbedding)

	// Keyword overlap bonus
	keywordBonus := ra.calculateKeywordOverlap(originalQuery, article)

	// Recency bonus (newer papers get slight boost)
	recencyBonus := ra.calculateRecencyBonus(article.PubDate)

	// Journal impact bonus (could be enhanced with actual impact factors)
	journalBonus := ra.calculateJournalBonus(article.Journal)

	// Combined score
	finalScore := cosineSim + (keywordBonus * 0.1) + (recencyBonus * 0.05) + (journalBonus * 0.05)

	return models.ArticleWithScore{
		Article:            article,
		Score:              finalScore,
		QueryRelevance:     keywordBonus,
		SemanticSimilarity: cosineSim,
	}
}

func (ra *ResearchAnalyzer) calculateKeywordOverlap(query string, article models.PubMedArticle) float64 {
	queryWords := strings.Fields(strings.ToLower(query))
	articleText := strings.ToLower(article.Title + " " + article.Abstract)

	matches := 0
	for _, word := range queryWords {
		if len(word) > 3 && strings.Contains(articleText, word) {
			matches++
		}
	}

	if len(queryWords) == 0 {
		return 0
	}
	return float64(matches) / float64(len(queryWords))
}

func (ra *ResearchAnalyzer) calculateRecencyBonus(pubDate string) float64 {
	// Simple recency bonus - newer papers get small boost
	if strings.Contains(pubDate, "2024") {
		return 0.1
	} else if strings.Contains(pubDate, "2023") {
		return 0.05
	}
	return 0.0
}

func (ra *ResearchAnalyzer) calculateJournalBonus(journal string) float64 {
	// High-impact journals (simplified)
	highImpactJournals := []string{
		"nature", "science", "cell", "lancet", "nejm", "jama",
		"nature medicine", "nature genetics", "plos medicine",
	}

	journalLower := strings.ToLower(journal)
	for _, highImpact := range highImpactJournals {
		if strings.Contains(journalLower, highImpact) {
			return 0.1
		}
	}
	return 0.0
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
