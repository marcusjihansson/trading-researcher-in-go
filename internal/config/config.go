package config

import (
	"fmt"
	"os"
)

// Configuration constants
const (
	GeminiAPIBase       = "https://generativelanguage.googleapis.com/v1beta"
	PubMedAPIBase       = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
	EmbeddingModel      = "embeddinggemma:300m"
	GenerationModel     = "models/gemini-2.5-flash"
	DefaultBackupModel  = "models/gemma-3-27b-it" // Default Gemma 3 27B as backup
	MaxResults          = 50                      // Increased for better embedding filtering
	TopArticles         = 8                       // More options for user
	SimilarityThreshold = 0.35                    // Lowered for finance domain and local embedding scales
)

// GetBackupModel returns the backup model from environment or default
func GetBackupModel() string {
	if model := os.Getenv("BACKUP_MODEL"); model != "" {
		return model
	}
	return DefaultBackupModel
}

// IsAPIEnabled checks if a specific API should be used
func IsAPIEnabled(apiName string) bool {
	envVar := "DISABLE_" + apiName
	return os.Getenv(envVar) == ""
}

// GetAPIKey retrieves API key from environment with helpful error messages
func GetAPIKey(envVar, apiName, signupURL string) (string, error) {
	key := os.Getenv(envVar)
	if key == "" {
		return "", fmt.Errorf("%s API key not found. Set %s environment variable or get a key from %s", apiName, envVar, signupURL)
	}
	return key, nil
}
