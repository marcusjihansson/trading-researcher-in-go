package clients

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"enhanced-research-analyzer/internal/config"
	"enhanced-research-analyzer/internal/models"
)

// convertToString converts interface{} to string, handling both string and number types
func convertToString(val interface{}) string {
	if val == nil {
		return ""
	}
	switch v := val.(type) {
	case string:
		return v
	case int:
		return strconv.Itoa(v)
	case int64:
		return strconv.FormatInt(v, 10)
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	default:
		return fmt.Sprintf("%v", v)
	}
}

// sanitizeText removes XML/HTML tags and trims whitespace
func sanitizeText(s string) string {
	if s == "" {
		return s
	}
	// Crossref abstracts can be JATS XML like <jats:p>...</jats:p>
	re := regexp.MustCompile(`<[^>]+>`) // naive tag stripper
	clean := re.ReplaceAllString(s, "")
	clean = strings.ReplaceAll(clean, "&nbsp;", " ")
	clean = strings.TrimSpace(clean)
	return clean
}

// SearchArxiv performs arXiv search
func SearchArxiv(ctx context.Context, query string, httpClient *http.Client) ([]models.PubMedArticle, error) {
	// Build arXiv query for quantitative finance categories
	cats := []string{"q-fin.TR", "q-fin.CP", "q-fin.ST", "q-fin.PR", "q-fin.RM", "q-fin.GN"}
	var catTerms []string
	for _, c := range cats {
		catTerms = append(catTerms, "cat:"+c)
	}
	searchExpr := fmt.Sprintf("all:%s AND (%s)", query, strings.Join(catTerms, " OR "))
	apiURL := fmt.Sprintf("http://export.arxiv.org/api/query?search_query=%s&start=0&max_results=%d&sortBy=relevance",
		url.QueryEscape(searchExpr), config.MaxResults)

	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create arXiv request: %w", err)
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to query arXiv: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("arXiv API error %d: %s", resp.StatusCode, string(b))
	}

	type arxivAuthor struct {
		Name string `xml:"name"`
	}
	type arxivEntry struct {
		ID        string        `xml:"id"`
		Title     string        `xml:"title"`
		Summary   string        `xml:"summary"`
		Published string        `xml:"published"`
		Authors   []arxivAuthor `xml:"author"`
	}
	type arxivFeed struct {
		Entries []arxivEntry `xml:"entry"`
	}

	var feed arxivFeed
	if err := xml.NewDecoder(resp.Body).Decode(&feed); err != nil {
		return nil, fmt.Errorf("failed to decode arXiv response: %w", err)
	}

	var results []models.PubMedArticle
	for _, e := range feed.Entries {
		if strings.TrimSpace(e.Title) == "" {
			continue
		}
		// Extract UID from arXiv ID URL
		uid := e.ID
		if idx := strings.LastIndex(uid, "/"); idx != -1 && idx+1 < len(uid) {
			uid = uid[idx+1:]
		}
		uid = "arxiv:" + uid
		var authors []models.Author
		for _, a := range e.Authors {
			authors = append(authors, models.Author{Name: a.Name})
		}
		results = append(results, models.PubMedArticle{
			UID:      uid,
			Title:    strings.TrimSpace(e.Title),
			Authors:  authors,
			Abstract: strings.TrimSpace(e.Summary),
			PubDate:  e.Published,
			Journal:  "arXiv",
			DOI:      e.ID, // use arXiv URL as link
		})
	}
	return results, nil
}

// SearchCrossref performs Crossref search
func SearchCrossref(ctx context.Context, query string, httpClient *http.Client) ([]models.PubMedArticle, error) {
	apiURL := fmt.Sprintf("https://api.crossref.org/works?rows=%d&sort=score&order=desc&filter=type:journal-article&query=%s", config.MaxResults, url.QueryEscape(query))
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create Crossref request: %w", err)
	}
	req.Header.Set("User-Agent", "enhanced-research-analyzer/1.0")

	// Add timeout specifically for Crossref (it can be slow)
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	req = req.WithContext(ctx)

	resp, err := httpClient.Do(req)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("Crossref API request timed out after 10 seconds")
		}
		return nil, fmt.Errorf("failed to query Crossref: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Crossref API error %d: %s", resp.StatusCode, string(b))
	}
	var cr struct {
		Message struct {
			Items []struct {
				Title          []string `json:"title"`
				ContainerTitle []string `json:"container-title"`
				Abstract       string   `json:"abstract"`
				DOI            string   `json:"DOI"`
				Author         []struct {
					Given  string `json:"given"`
					Family string `json:"family"`
				} `json:"author"`
				Issued struct {
					DateParts [][]int `json:"date-parts"`
				} `json:"issued"`
			} `json:"items"`
		} `json:"message"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return nil, fmt.Errorf("failed to decode Crossref response: %w", err)
	}
	var results []models.PubMedArticle
	for _, it := range cr.Message.Items {
		if len(it.Title) == 0 || strings.TrimSpace(it.Title[0]) == "" {
			continue
		}
		var authors []models.Author
		for _, a := range it.Author {
			name := strings.TrimSpace(strings.TrimSpace(a.Given + " " + a.Family))
			if name != "" {
				authors = append(authors, models.Author{Name: name})
			}
		}
		pubDate := ""
		if len(it.Issued.DateParts) > 0 && len(it.Issued.DateParts[0]) > 0 {
			pubDate = fmt.Sprintf("%d", it.Issued.DateParts[0][0])
		}
		journal := ""
		if len(it.ContainerTitle) > 0 {
			journal = it.ContainerTitle[0]
		}
		abstract := sanitizeText(it.Abstract)
		uid := it.DOI
		if uid != "" {
			uid = "doi:" + uid
		} else {
			uid = "crossref:" + strings.ToLower(strings.ReplaceAll(it.Title[0], " ", "-"))
		}
		results = append(results, models.PubMedArticle{
			UID:      uid,
			Title:    strings.TrimSpace(it.Title[0]),
			Authors:  authors,
			Abstract: strings.TrimSpace(abstract),
			PubDate:  pubDate,
			Journal:  journal,
			DOI:      it.DOI,
		})
	}
	return results, nil
}

// SearchSemanticScholar performs Semantic Scholar search
func SearchSemanticScholar(ctx context.Context, query string, httpClient *http.Client) ([]models.PubMedArticle, error) {
	apiURL := fmt.Sprintf("https://api.semanticscholar.org/graph/v1/paper/search?query=%s&limit=%d&fields=title,abstract,authors,year,venue,externalIds,url", url.QueryEscape(query), config.MaxResults)
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create Semantic Scholar request: %w", err)
	}

	// Add API key if available for higher rate limits
	if key := strings.TrimSpace(os.Getenv("S2_API_KEY")); key != "" {
		req.Header.Set("x-api-key", key)
	} else {
		// Free tier - add polite headers to avoid being blocked
		req.Header.Set("User-Agent", "enhanced-research-analyzer/1.0")
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to query Semantic Scholar: %w", err)
	}
	defer resp.Body.Close()

	// Handle rate limiting gracefully
	if resp.StatusCode == 429 {
		return nil, fmt.Errorf("Semantic Scholar API rate limit exceeded. Consider getting an API key from https://www.semanticscholar.org/product/api#api-key-form or wait before retrying")
	}

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Semantic Scholar API error %d: %s", resp.StatusCode, string(b))
	}
	var ss struct {
		Total int `json:"total"`
		Data  []struct {
			Title       string            `json:"title"`
			Abstract    string            `json:"abstract"`
			Year        int               `json:"year"`
			Venue       string            `json:"venue"`
			URL         string            `json:"url"`
			ExternalIds map[string]string `json:"externalIds"`
			Authors     []struct {
				Name string `json:"name"`
			} `json:"authors"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&ss); err != nil {
		return nil, fmt.Errorf("failed to decode Semantic Scholar response: %w", err)
	}
	var results []models.PubMedArticle
	for _, p := range ss.Data {
		if strings.TrimSpace(p.Title) == "" {
			continue
		}
		var authors []models.Author
		for _, a := range p.Authors {
			if strings.TrimSpace(a.Name) != "" {
				authors = append(authors, models.Author{Name: a.Name})
			}
		}
		pubDate := ""
		if p.Year > 0 {
			pubDate = fmt.Sprintf("%d", p.Year)
		}
		doi := p.ExternalIds["DOI"]
		uid := doi
		if uid == "" {
			uid = "s2:" + strings.ToLower(strings.ReplaceAll(p.Title, " ", "-"))
		} else {
			uid = "doi:" + uid
		}
		journal := p.Venue
		if journal == "" {
			journal = "Semantic Scholar"
		}
		results = append(results, models.PubMedArticle{
			UID:      uid,
			Title:    strings.TrimSpace(p.Title),
			Authors:  authors,
			Abstract: strings.TrimSpace(p.Abstract),
			PubDate:  pubDate,
			Journal:  journal,
			DOI:      doi,
		})
	}
	return results, nil
}

// SearchCoreAPI performs CORE API search
func SearchCoreAPI(ctx context.Context, query string, httpClient *http.Client) ([]models.PubMedArticle, error) {
	if !config.IsAPIEnabled("CORE") {
		return nil, fmt.Errorf("CORE API disabled via DISABLE_CORE environment variable")
	}

	apiKey, err := config.GetAPIKey("CORE_API", "CORE", "https://core.ac.uk/services/api/")
	if err != nil {
		return nil, err
	}

	// CORE v3 search works endpoint
	apiURL := fmt.Sprintf("https://api.core.ac.uk/v3/search/works?q=%s&page=1&page_size=%d", url.QueryEscape(query), config.MaxResults)
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create CORE API request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Accept", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to query CORE API: %w", err)
	}
	defer resp.Body.Close()

	// Handle rate limiting
	if resp.StatusCode == 429 {
		return nil, fmt.Errorf("CORE API rate limit exceeded - please wait before retrying")
	}

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("CORE API error %d: %s", resp.StatusCode, string(b))
	}

	// Response shape is loosely modeled to be resilient to missing fields
	var coreResp struct {
		Results []struct {
			ID        interface{} `json:"id"` // Can be string or number
			Title     string      `json:"title"`
			Abstract  string      `json:"abstract"`
			Year      int         `json:"year"`
			Publisher string      `json:"publisher"`
			DOI       string      `json:"doi"`
			Authors   []struct {
				Name string `json:"name"`
			} `json:"authors"`
			URLs []string `json:"urls"`
		} `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&coreResp); err != nil {
		return nil, fmt.Errorf("failed to decode CORE API response: %w", err)
	}

	var results []models.PubMedArticle
	for _, w := range coreResp.Results {
		if strings.TrimSpace(w.Title) == "" {
			continue
		}
		var authors []models.Author
		for _, a := range w.Authors {
			if strings.TrimSpace(a.Name) != "" {
				authors = append(authors, models.Author{Name: a.Name})
			}
		}
		pubDate := ""
		if w.Year > 0 {
			pubDate = fmt.Sprintf("%d", w.Year)
		}

		// Convert ID from interface{} to string (handles both string and number)
		idStr := convertToString(w.ID)
		uid := w.DOI
		if uid == "" {
			if idStr != "" {
				uid = "core:" + idStr
			} else {
				uid = "core:" + strings.ToLower(strings.ReplaceAll(w.Title, " ", "-"))
			}
		} else {
			uid = "doi:" + uid
		}
		journal := w.Publisher
		if journal == "" {
			journal = "CORE"
		}
		link := w.DOI
		if link == "" && len(w.URLs) > 0 {
			link = w.URLs[0]
		}
		results = append(results, models.PubMedArticle{
			UID:      uid,
			Title:    strings.TrimSpace(w.Title),
			Authors:  authors,
			Abstract: strings.TrimSpace(w.Abstract),
			PubDate:  pubDate,
			Journal:  journal,
			DOI:      link,
		})
	}
	return results, nil
}
