package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"enhanced-research-analyzer/internal/models"
	"enhanced-research-analyzer/internal/services"
	"enhanced-research-analyzer/internal/utils"
)

func main() {
	// Get API key from environment variable
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("❌ Error: Please set GEMINI_API_KEY environment variable")
		fmt.Println("You can get an API key from: https://ai.google.dev/")
		os.Exit(1)
	}

	fmt.Println("🔬 Enhanced Research Paper Analyzer with DSPy-Go + Advanced Embeddings")
	fmt.Println("========================================================================")
	fmt.Println("🚀 Features: Query Enhancement | Advanced Embeddings | Smart Ranking")
	fmt.Println("🧠 Powered by: Gemini Flash + Embeddings + DSPy-Inspired Optimization")

	analyzer := services.NewResearchAnalyzer(apiKey)
	ctx := context.Background()

	// Optional: Load optimization examples (in a real system, this would come from user feedback)
	analyzer.OptimizationData = []models.Example{
		{
			Inputs: map[string]interface{}{
				"original_query": "AI in healthcare",
			},
			Outputs: map[string]interface{}{
				"response": `{
					"expanded_queries": ["artificial intelligence medical diagnosis", "machine learning clinical decision support", "deep learning healthcare applications"],
					"key_concepts": ["neural networks", "predictive modeling", "clinical AI"],
					"related_terms": ["computer-aided diagnosis", "medical imaging AI", "electronic health records"]
				}`,
			},
		},
	}

	// Optimize modules with example data
	analyzer.QueryEnhancer.Optimize(analyzer.OptimizationData)
	analyzer.AnalysisModule.Optimize(analyzer.OptimizationData)

	fmt.Println("\n✅ System initialized with DSPy optimization examples")

	for {
		fmt.Println("\n" + strings.Repeat("=", 60))

		// Get research query from user
		query := utils.GetUserInput("\n🔍 Enter your research query (or 'quit' to exit): ")
		if strings.ToLower(query) == "quit" {
			fmt.Println("👋 Thanks for using the Enhanced Research Analyzer!")
			break
		}

		if query == "" {
			fmt.Println("❌ Please enter a valid query.")
			continue
		}

		startTime := time.Now()

		// Enhanced Step 1: Search with query enhancement
		articles, enhancedQuery, err := analyzer.EnhancedSearch(ctx, query)
		if err != nil {
			fmt.Printf("❌ Error in enhanced search: %v\n", err)
			continue
		}

		if len(articles) == 0 {
			fmt.Println("❌ No articles found. Try broader search terms.")
			continue
		}

		// Enhanced Step 2: Advanced similarity analysis
		relevantArticles, err := analyzer.FindRelevantArticles(ctx, enhancedQuery, articles)
		if err != nil {
			fmt.Printf("❌ Error in relevance analysis: %v\n", err)
			continue
		}

		if len(relevantArticles) == 0 {
			fmt.Printf("❌ No highly relevant articles found (similarity threshold: %.1f).\n", 0.35) // TODO: use config
			fmt.Println("💡 Try different search terms or check back later for new publications.")
			continue
		}

		// Display enhanced results
		utils.DisplayEnhancedArticles(relevantArticles, enhancedQuery)

		// Performance metrics
		elapsed := time.Since(startTime)
		fmt.Printf("\n⚡ Analysis completed in %.2f seconds\n", elapsed.Seconds())
		fmt.Printf("📊 Processed %d total articles, found %d highly relevant matches\n",
			len(articles), len(relevantArticles))

		// Article selection loop - stays within the same search results
		for {
			// Enhanced Step 3: Article selection with additional options
			fmt.Println("\n🎯 Selection Options:")
			fmt.Println("   1-" + strconv.Itoa(len(relevantArticles)) + ": Analyze specific article")
			fmt.Println("   'summary': Get quick summary of all top articles")
			fmt.Println("   'concepts': Show key concepts found")
			fmt.Println("   'new': Start new search")

			selection := utils.GetUserInput("\n📖 Your choice: ")

			// Handle different selection types
			switch strings.ToLower(selection) {
			case "summary":
				fmt.Println("\n📋 QUICK SUMMARY OF TOP ARTICLES")
				fmt.Println(strings.Repeat("=", 50))
				for i, item := range relevantArticles[:utils.Min(3, len(relevantArticles))] {
					fmt.Printf("\n%d. %s\n", i+1, item.Article.Title)
					fmt.Printf("   Score: %.3f | %s (%s)\n", item.Score, item.Article.Journal, item.Article.PubDate)
					if len(item.Article.Abstract) > 0 {
						abstract := item.Article.Abstract
						if len(abstract) > 150 {
							abstract = abstract[:150] + "..."
						}
						fmt.Printf("   %s\n", abstract)
					}
				}
				continue

			case "concepts":
				fmt.Println("\n🧠 KEY CONCEPTS IDENTIFIED")
				fmt.Println(strings.Repeat("=", 40))
				fmt.Printf("Original Query: %s\n", enhancedQuery.Original)
				if len(enhancedQuery.Expanded) > 0 {
					fmt.Printf("Enhanced Queries: %s\n", strings.Join(enhancedQuery.Expanded, " | "))
				}
				if len(enhancedQuery.Concepts) > 0 {
					fmt.Printf("Related Concepts: %s\n", strings.Join(enhancedQuery.Concepts, " | "))
				}
				continue

			case "new":
				break

			default:
				// Try to parse as article number
				index, err := strconv.Atoi(selection)
				if err != nil || index < 1 || index > len(relevantArticles) {
					fmt.Println("❌ Invalid selection. Please try again.")
					continue
				}

				selectedArticle := relevantArticles[index-1]
				fmt.Printf("\n✅ Selected: %s\n", selectedArticle.Article.Title)
				fmt.Printf("🎯 Relevance Score: %.3f\n\n", selectedArticle.Score)

				// Enhanced Step 4: Advanced analysis using DSPy module
				analysisStart := time.Now()
				analysis, err := analyzer.AnalyzeArticle(ctx, selectedArticle.Article, query)
				if err != nil {
					fmt.Printf("❌ Error analyzing article: %v\n", err)
					continue
				}

				analysisTime := time.Since(analysisStart)

				// Display enhanced analysis
				fmt.Println("📊 COMPREHENSIVE ANALYSIS")
				fmt.Println(strings.Repeat("=", 90))
				fmt.Println(analysis)
				fmt.Println(strings.Repeat("=", 90))

				fmt.Printf("\n⚡ Analysis generated in %.2f seconds\n", analysisTime.Seconds())

				// Offer PDF/link option
				if link := utils.BestArticleLink(selectedArticle.Article); link != "" {
					want := utils.GetUserInput("\n📎 Do you want a link to the PDF (or DOI/URL)? (y/N): ")
					if strings.ToLower(strings.TrimSpace(want)) == "y" {
						fmt.Printf("\n🔗 Link: %s\n", link)
					}
				}

				// Ask if user wants to analyze another article or search again
				nextAction := utils.GetUserInput("\n🔄 Next action? ('another' for different article, 'ask' for follow-up questions, 'search' for new query, Enter to continue): ")
				if strings.ToLower(nextAction) == "ask" || strings.ToLower(nextAction) == "followup" || strings.ToLower(nextAction) == "question" {
					// Handle follow-up questions about the article
					question := utils.GetUserInput("\n❓ What question do you have about this article? ")
					if strings.TrimSpace(question) != "" {
						fmt.Println("\n🤔 Thinking about your question...")
						followupStart := time.Now()

						// Use the follow-up question module
						followupAnswer, err := analyzer.AnswerFollowupQuestion(ctx, selectedArticle.Article, question)
						if err != nil {
							fmt.Printf("❌ Error answering question: %v\n", err)
						} else {
							followupTime := time.Since(followupStart)
							fmt.Println("\n💡 FOLLOW-UP ANSWER")
							fmt.Println(strings.Repeat("=", 60))
							fmt.Println(followupAnswer)
							fmt.Println(strings.Repeat("=", 60))
							fmt.Printf("\n⚡ Answer generated in %.2f seconds\n", followupTime.Seconds())
						}
					}
					// After answering, continue with the same article options
					continue
				} else if strings.ToLower(nextAction) == "another" {
					utils.DisplayEnhancedArticles(relevantArticles, enhancedQuery)
					// Continue the inner loop to show selection options again
					continue
				} else if strings.ToLower(nextAction) == "search" {
					break
				}
			}
			// If we get here, break out of the article selection loop
			break
		}
	}
}
