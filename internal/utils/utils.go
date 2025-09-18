package utils

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"

	"enhanced-research-analyzer/internal/models"
)

// Utility functions

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

func FormatAuthors(authors []models.Author) string {
	var names []string
	for _, author := range authors {
		names = append(names, author.Name)
	}
	if len(names) > 3 {
		return strings.Join(names[:3], ", ") + ", et al."
	}
	return strings.Join(names, ", ")
}

// bestArticleLink returns the best available link for the article.
// Preference order:
// 1) arXiv PDF when UID starts with "arxiv:"
// 2) https://doi.org/<DOI> if DOI looks like a bare DOI (10.x/..)
// 3) If DOI is already a URL, return it; convert arXiv abs to pdf
func BestArticleLink(a models.PubMedArticle) string {
	uid := strings.ToLower(strings.TrimSpace(a.UID))
	if strings.HasPrefix(uid, "arxiv:") {
		id := strings.TrimPrefix(uid, "arxiv:")
		if id != "" {
			return "https://arxiv.org/pdf/" + id + ".pdf"
		}
	}
	doi := strings.TrimSpace(a.DOI)
	if doi == "" {
		return ""
	}
	low := strings.ToLower(doi)
	if strings.HasPrefix(low, "doi:") {
		doi = strings.TrimSpace(doi[4:])
		low = strings.ToLower(doi)
	}
	if strings.HasPrefix(doi, "10.") {
		return "https://doi.org/" + doi
	}
	if strings.HasPrefix(low, "http://") || strings.HasPrefix(low, "https://") {
		if strings.Contains(low, "arxiv.org/abs/") {
			pdf := strings.Replace(low, "/abs/", "/pdf/", 1)
			if !strings.HasSuffix(pdf, ".pdf") {
				pdf += ".pdf"
			}
			return pdf
		}
		return a.DOI
	}
	return ""
}

func DisplayEnhancedArticles(articles []models.ArticleWithScore, enhancedQuery *models.EnhancedQuery) {
	fmt.Printf("📚 Top Relevant Articles for: \"%s\"\n", enhancedQuery.Original)
	if len(enhancedQuery.Expanded) > 0 {
		fmt.Printf("💡 Enhanced with: %s\n", strings.Join(enhancedQuery.Expanded[:min(2, len(enhancedQuery.Expanded))], ", "))
	}
	fmt.Println(strings.Repeat("=", 90))

	for i, item := range articles {
		article := item.Article
		fmt.Printf("\n[%d] %s\n", i+1, article.Title)
		fmt.Printf("    👥 Authors: %s\n", FormatAuthors(article.Authors))
		fmt.Printf("    📖 Journal: %s (%s)\n", article.Journal, article.PubDate)
		fmt.Printf("    🎯 Overall Score: %.3f | Semantic: %.3f | Keywords: %.3f\n",
			item.Score, item.SemanticSimilarity, item.QueryRelevance)
		if article.DOI != "" {
			fmt.Printf("    🔗 DOI: %s\n", article.DOI)
		}
		if len(article.Abstract) > 0 {
			abstract := article.Abstract
			if len(abstract) > 300 {
				abstract = abstract[:300] + "..."
			}
			fmt.Printf("    📄 Abstract: %s\n", abstract)
		}
		fmt.Println(strings.Repeat("-", 90))
	}
}

func GetUserInput(prompt string) string {
	fmt.Print(prompt)
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	return strings.TrimSpace(scanner.Text())
}

func Min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
