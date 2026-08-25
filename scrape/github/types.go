package github

// RepoSearchResponse is the top-level response from the GitHub repository search API.
type RepoSearchResponse struct {
	TotalCount        int        `json:"total_count"`
	IncompleteResults bool       `json:"incomplete_results"`
	Items             []RepoItem `json:"items"`
}

// RepoItem represents a single repository in the search results.
type RepoItem struct {
	ID                        int          `json:"id"`
	NodeID                    string       `json:"node_id"`
	Name                      string       `json:"name"`
	FullName                  string       `json:"full_name"`
	Owner                     *Owner       `json:"owner"`
	Private                   bool         `json:"private"`
	HTMLURL                   string       `json:"html_url"`
	Description               *string      `json:"description"`
	Fork                      bool         `json:"fork"`
	URL                       string       `json:"url"`
	CreatedAt                 string       `json:"created_at"`
	UpdatedAt                 string       `json:"updated_at"`
	PushedAt                  string       `json:"pushed_at"`
	Homepage                  *string      `json:"homepage"`
	Size                      int          `json:"size"`
	StargazersCount           int          `json:"stargazers_count"`
	WatchersCount             int          `json:"watchers_count"`
	Language                  *string      `json:"language"`
	ForksCount                int          `json:"forks_count"`
	MasterBranch              string       `json:"master_branch"`
	DefaultBranch             string       `json:"default_branch"`
	Score                     float64      `json:"score"`
	ForksURL                  string       `json:"forks_url"`
	KeysURL                   string       `json:"keys_url"`
	CollaboratorsURL          string       `json:"collaborators_url"`
	TeamsURL                  string       `json:"teams_url"`
	HooksURL                  string       `json:"hooks_url"`
	IssueEventsURL            string       `json:"issue_events_url"`
	EventsURL                 string       `json:"events_url"`
	AssigneesURL              string       `json:"assignees_url"`
	BranchesURL               string       `json:"branches_url"`
	TagsURL                   string       `json:"tags_url"`
	BlobsURL                  string       `json:"blobs_url"`
	GitTagsURL                string       `json:"git_tags_url"`
	GitRefsURL                string       `json:"git_refs_url"`
	TreesURL                  string       `json:"trees_url"`
	StatusesURL               string       `json:"statuses_url"`
	LanguagesURL              string       `json:"languages_url"`
	StargazersURL             string       `json:"stargazers_url"`
	ContributorsURL           string       `json:"contributors_url"`
	SubscribersURL            string       `json:"subscribers_url"`
	SubscriptionURL           string       `json:"subscription_url"`
	CommitsURL                string       `json:"commits_url"`
	GitCommitsURL             string       `json:"git_commits_url"`
	CommentsURL               string       `json:"comments_url"`
	IssueCommentURL           string       `json:"issue_comment_url"`
	ContentsURL               string       `json:"contents_url"`
	CompareURL                string       `json:"compare_url"`
	MergesURL                 string       `json:"merges_url"`
	ArchiveURL                string       `json:"archive_url"`
	DownloadsURL              string       `json:"downloads_url"`
	IssuesURL                 string       `json:"issues_url"`
	PullsURL                  string       `json:"pulls_url"`
	MilestonesURL             string       `json:"milestones_url"`
	NotificationsURL          string       `json:"notifications_url"`
	LabelsURL                 string       `json:"labels_url"`
	ReleasesURL               string       `json:"releases_url"`
	DeploymentsURL            string       `json:"deployments_url"`
	GitURL                    string       `json:"git_url"`
	SSHURL                    string       `json:"ssh_url"`
	CloneURL                  string       `json:"clone_url"`
	SVNURL                    string       `json:"svn_url"`
	Forks                     int          `json:"forks"`
	OpenIssues                int          `json:"open_issues"`
	Watchers                  int          `json:"watchers"`
	Topics                    []string     `json:"topics"`
	MirrorURL                 *string      `json:"mirror_url"`
	HasIssues                 bool         `json:"has_issues"`
	HasProjects               bool         `json:"has_projects"`
	HasPages                  bool         `json:"has_pages"`
	HasWiki                   bool         `json:"has_wiki"`
	HasDownloads              bool         `json:"has_downloads"`
	HasDiscussions            bool         `json:"has_discussions"`
	HasPullRequests           bool         `json:"has_pull_requests"`
	PullRequestCreationPolicy string       `json:"pull_request_creation_policy"`
	Archived                  bool         `json:"archived"`
	Disabled                  bool         `json:"disabled"`
	Visibility                string       `json:"visibility"`
	License                   *License     `json:"license"`
	Permissions               *Permissions `json:"permissions"`
	TextMatches               []TextMatch  `json:"text_matches"`
	TempCloneToken            string       `json:"temp_clone_token"`
	AllowMergeCommit          bool         `json:"allow_merge_commit"`
	AllowSquashMerge          bool         `json:"allow_squash_merge"`
	AllowRebaseMerge          bool         `json:"allow_rebase_merge"`
	AllowAutoMerge            bool         `json:"allow_auto_merge"`
	DeleteBranchOnMerge       bool         `json:"delete_branch_on_merge"`
	AllowForking              bool         `json:"allow_forking"`
	IsTemplate                bool         `json:"is_template"`
	WebCommitSignoffRequired  bool         `json:"web_commit_signoff_required"`
}

// Owner is the Simple User object embedded in a RepoItem.
type Owner struct {
	Name              *string `json:"name"`
	Email             *string `json:"email"`
	Login             string  `json:"login"`
	ID                int64   `json:"id"`
	NodeID            string  `json:"node_id"`
	AvatarURL         string  `json:"avatar_url"`
	GravatarID        *string `json:"gravatar_id"`
	URL               string  `json:"url"`
	HTMLURL           string  `json:"html_url"`
	FollowersURL      string  `json:"followers_url"`
	FollowingURL      string  `json:"following_url"`
	GistsURL          string  `json:"gists_url"`
	StarredURL        string  `json:"starred_url"`
	SubscriptionsURL  string  `json:"subscriptions_url"`
	OrganizationsURL  string  `json:"organizations_url"`
	ReposURL          string  `json:"repos_url"`
	EventsURL         string  `json:"events_url"`
	ReceivedEventsURL string  `json:"received_events_url"`
	Type              string  `json:"type"`
	SiteAdmin         bool    `json:"site_admin"`
	StarredAt         string  `json:"starred_at"`
	UserViewType      string  `json:"user_view_type"`
}

// License is the License Simple object embedded in a RepoItem.
type License struct {
	Key     string  `json:"key"`
	Name    string  `json:"name"`
	URL     *string `json:"url"`
	SPDXID  *string `json:"spdx_id"`
	NodeID  string  `json:"node_id"`
	HTMLURL string  `json:"html_url"`
}

// Permissions represents access permissions on a repository.
type Permissions struct {
	Admin    bool `json:"admin"`
	Maintain bool `json:"maintain"`
	Push     bool `json:"push"`
	Triage   bool `json:"triage"`
	Pull     bool `json:"pull"`
}

// TextMatch captures a single text match result within a search response.
type TextMatch struct {
	ObjectURL  string  `json:"object_url"`
	ObjectType *string `json:"object_type"`
	Property   string  `json:"property"`
	Fragment   string  `json:"fragment"`
	Matches    []Match `json:"matches"`
}

// Match represents an individual matched substring within a fragment.
type Match struct {
	Text    string `json:"text"`
	Indices []int  `json:"indices"`
}

// CommitResponse is a single commit from the GitHub API.
type CommitResponse struct {
	SHA         string         `json:"sha"`
	NodeID      string         `json:"node_id"`
	Commit      CommitDetail   `json:"commit"`
	URL         string         `json:"url"`
	HTMLURL     string         `json:"html_url"`
	CommentsURL string         `json:"comments_url"`
	Author      *Owner         `json:"author"`
	Committer   *Owner         `json:"committer"`
	Parents     []CommitParent `json:"parents"`
	Stats       *CommitStats   `json:"stats"`
	Files       []CommitFile   `json:"files"`
}

// CommitDetail holds git-level metadata inside a CommitResponse.
type CommitDetail struct {
	Author       GitSignature `json:"author"`
	Committer    GitSignature `json:"committer"`
	Message      string       `json:"message"`
	Tree         TreeInfo     `json:"tree"`
	URL          string       `json:"url"`
	CommentCount int          `json:"comment_count"`
	Verification Verification `json:"verification"`
}

// GitSignature represents a git author/committer signature.
type GitSignature struct {
	Name  string `json:"name"`
	Email string `json:"email"`
	Date  string `json:"date"`
}

// TreeInfo represents a git tree reference.
type TreeInfo struct {
	SHA string `json:"sha"`
	URL string `json:"url"`
}

// Verification represents commit signature verification status.
type Verification struct {
	Verified   bool    `json:"verified"`
	Reason     string  `json:"reason"`
	Signature  *string `json:"signature"`
	Payload    *string `json:"payload"`
	VerifiedAt *string `json:"verified_at"`
}

// CommitParent represents a parent commit reference.
type CommitParent struct {
	SHA     string `json:"sha"`
	URL     string `json:"url"`
	HTMLURL string `json:"html_url"`
}

// CommitStats represents line statistics for a commit.
type CommitStats struct {
	Total     int `json:"total"`
	Additions int `json:"additions"`
	Deletions int `json:"deletions"`
}

// CommitFile represents a file changed in a commit.
type CommitFile struct {
	SHA              string  `json:"sha"`
	Filename         string  `json:"filename"`
	Status           string  `json:"status"`
	Additions        int     `json:"additions"`
	Deletions        int     `json:"deletions"`
	Changes          int     `json:"changes"`
	BlobURL          string  `json:"blob_url"`
	RawURL           string  `json:"raw_url"`
	ContentsURL      string  `json:"contents_url"`
	Patch            *string `json:"patch"`
	PreviousFilename *string `json:"previous_filename"`
}

// TreeResponse is the top-level response from the GitHub git trees API.
type TreeResponse struct {
	SHA       string      `json:"sha"`
	URL       string      `json:"url"`
	Tree      []TreeEntry `json:"tree"`
	Truncated bool        `json:"truncated"`
}

// TreeEntry represents a single object inside a git tree.
type TreeEntry struct {
	Path string `json:"path"`
	Mode string `json:"mode"`
	Type string `json:"type"`
	SHA  string `json:"sha"`
	Size *int   `json:"size"` // only present for blobs
	URL  string `json:"url"`
}

// BlobResponse is the top-level response from the GitHub git blobs API.
type BlobResponse struct {
	SHA      string `json:"sha"`
	NodeID   string `json:"node_id"`
	Size     int    `json:"size"`
	URL      string `json:"url"`
	Content  string `json:"content"`
	Encoding string `json:"encoding"`
}

type CommitInfo struct {
	Sha string `json:"sha"`
	Url string `json:"url"`
}

type BranchInfo struct {
	Name      string     `json:"name"`
	Commit    CommitInfo `json:"commit"`
	Protected bool       `json:"protected"`
}
