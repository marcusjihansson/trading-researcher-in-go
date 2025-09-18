package models

import "context"

// DSPy-Go inspired structures
type DSPySignature struct {
	InputFields  []string
	OutputFields []string
	Description  string
}

type DSPyModule interface {
	Forward(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error)
	GetSignature() DSPySignature
	Optimize(examples []Example) error
}

type Example struct {
	Inputs  map[string]interface{}
	Outputs map[string]interface{}
}

// Enhanced structures
type PubMedSearchResult struct {
	ESummaryResult struct {
		Result map[string]PubMedArticle `json:"result"`
	} `json:"result"`
}

type PubMedArticle struct {
	UID         string   `json:"uid"`
	Title       string   `json:"title"`
	Authors     []Author `json:"authors"`
	Abstract    string   `json:"abstract"`
	PubDate     string   `json:"pubdate"`
	Journal     string   `json:"source"`
	DOI         string   `json:"elocationid"`
	SortPubDate string   `json:"sortpubdate"`
	Keywords    []string `json:"keywords,omitempty"`
}

type Author struct {
	Name     string `json:"name"`
	AuthType string `json:"authtype"`
}

type GeminiEmbeddingRequest struct {
	Model   string `json:"model"`
	Content struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"content"`
	TaskType string `json:"taskType,omitempty"`
}

type GeminiEmbeddingResponse struct {
	Embedding struct {
		Values []float64 `json:"values"`
	} `json:"embedding"`
}

type GeminiGenerateRequest struct {
	Contents []struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"contents"`
	GenerationConfig struct {
		Temperature     float64 `json:"temperature"`
		TopK            int     `json:"topK"`
		TopP            float64 `json:"topP"`
		MaxOutputTokens int     `json:"maxOutputTokens"`
	} `json:"generationConfig"`
	SystemInstruction *struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"systemInstruction,omitempty"`
}

type GeminiGenerateResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

type ArticleWithScore struct {
	Article            PubMedArticle
	Score              float64
	QueryRelevance     float64
	SemanticSimilarity float64
}

type EnhancedQuery struct {
	Original   string
	Expanded   []string
	Embeddings []float64
	Concepts   []string
}
