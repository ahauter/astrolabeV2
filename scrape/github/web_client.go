package github

import (
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"time"
)

type RetryRoundTripper struct {
	apiKey     string
	transport  http.RoundTripper
	maxRetries int
	backoff    time.Duration
	logger     *slog.Logger
}

func (rrt *RetryRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	logger := rrt.logger
	if logger == nil {
		logger = slog.Default()
	}
	if rrt.apiKey != "" {
		authHeader := fmt.Sprintf("Bearer %s", rrt.apiKey)
		req.Header.Set("Authorization", authHeader)
	}
	for i := 0; i <= rrt.maxRetries; i++ {
		newReq := req.Clone(req.Context())
		resp, err := rrt.transport.RoundTrip(newReq)
		backoff := rrt.backoff
		if err == nil {
			if resp.StatusCode == http.StatusOK {
				if i > 0 {
					logger.Info("request succeeded after retry", "url", req.URL.String(), "attempts", i+1)
				}
				return resp, err
			}
			if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode == 403 {
				if rateLim, exists := resp.Header["X-RateLimit-Reset"]; exists {
					reset_time, err := strconv.Atoi(rateLim[0])
					if err != nil {
						logger.Error("failed to parse rate limit reset header", "value", rateLim[0])
						panic("Error parsing int response in rate limit header, exiting")
					}
					backoff = time.Until(time.Unix(int64(reset_time), 0))
					logger.Warn("rate limited", "url", req.URL.String(), "reset_in", backoff.String(), "attempt", i+1)
				} else {
					logger.Warn("retrying after error", "url", req.URL.String(), "status", resp.StatusCode, "attempt", i+1)
				}
			} else {
				logger.Warn("retrying after error", "url", req.URL.String(), "status", resp.StatusCode, "attempt", i+1)
			}
		} else {
			logger.Warn("retrying after error", "url", req.URL.String(), "error", err, "attempt", i+1)
		}
		tmr := time.NewTimer(backoff)
		select {
		case <-req.Context().Done():
			return nil, errors.New("Context cancelled before request can be completed")
			// retry after backoff
		case <-tmr.C:
		}
	}
	logger.Error("max retries exceeded", "url", req.URL.String(), "max_retries", rrt.maxRetries)
	return nil, errors.New("Hit max retries with request")
}

func NewRetryRoundTripper(
	max_retries int, api_key string, backoff time.Duration, logger *slog.Logger,
) RetryRoundTripper {
	return RetryRoundTripper{
		transport:  &http.Transport{},
		apiKey:     api_key,
		maxRetries: max_retries,
		backoff:    backoff,
		logger:     logger,
	}
}

// constructs http client with RetryRoundTripper with sensible defaults
func NewClient(api_key string, logger *slog.Logger) http.Client {
	rrt := NewRetryRoundTripper(10, api_key, time.Minute, logger)
	return http.Client{
		Transport: &rrt,
		Timeout:   10 * time.Second,
	}
}
