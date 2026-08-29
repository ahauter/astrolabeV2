package github

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"time"
)

type RetryRoundTripper struct {
	apiKey     string
	transport  http.RoundTripper
	maxRetries int
	backoff    time.Duration
}

func (rrt *RetryRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if rrt.apiKey != "" {
		authHeader := fmt.Sprintf("Bearer: %s", rrt.apiKey)
		req.Header.Set("Authorization", authHeader)
	}
	for i := 0; i <= rrt.maxRetries; i++ {
		newReq := req.Clone(context.Background())
		resp, err := rrt.transport.RoundTrip(newReq)
		backoff := rrt.backoff
		if err == nil {
			if resp.StatusCode == http.StatusOK {
				return resp, err
			}
			if resp.StatusCode == http.StatusTooManyRequests {
				if rateLim, exists := resp.Header["X-RateLimit-Reset"]; exists {
					reset_time, err := strconv.Atoi(rateLim[0])
					if err != nil {
						panic("Error parsing int response in rate limit header, exiting")
					}
					backoff = time.Until(time.Unix(int64(reset_time), 0))
				}
			}
		}
		// retry after backoff
		time.Sleep(backoff)
	}
	return nil, errors.New("Hit max retries with request")
}

func NewRetryRoundTripper(
	max_retries int, api_key string, backoff time.Duration,
) RetryRoundTripper {
	return RetryRoundTripper{
		transport:  &http.Transport{},
		apiKey:     api_key,
		maxRetries: max_retries,
		backoff:    backoff,
	}
}

// constructs http client with RetryRoundTripper with sensible defaults
func NewClient(api_key string) http.Client {
	rrt := NewRetryRoundTripper(10, api_key, time.Minute)
	return http.Client{
		Transport: &rrt,
		Timeout:   10 * time.Second,
	}
}
