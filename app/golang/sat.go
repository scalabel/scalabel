package main

// TODO
type DashboardContents struct {
	Tasks      []Task      `json:"tasks"`
	VideoTasks []VideoTask `json:"videoTasks"`
}

// An annotation task to be completed by a user.
type Task struct {
	AssignmentID      string       `json:"assignmentId"`
	ProjectName       string       `json:"projectName"`
	WorkerID          string       `json:"workerId"`
	Category          []string     `json:"category"`
	LabelType         string       `json:"labelType"`
	TaskSize          int          `json:"taskSize"`
	Items             []ItemObject `json:"items"`
	SubmitTime        int64        `json:"submitTime"`
	NumSubmissions    int          `json:"numSubmissions"`
	NumLabeledItems   int          `json:"numLabeledItems"`
	NumDisplayedItems int          `json:"numDisplayedItems"`
	StartTime         int64        `json:"startTime"`
	Events            []Event      `json:"events"`
	VendorID          string       `json:"vendorId"`
	IPAddress         interface{}  `json:"ipAddress"`
	UserAgent         string       `json:"userAgent"`
}

// A result containing a list of items.
type Result struct {
	Items []ItemObject `json:"items"`
}

// Info pertaining to a task.
type TaskInfo struct {
	AssignmentID    string `json:"assignmentId"`
	ProjectName     string `json:"projectName"`
	WorkerID        string `json:"workerId"`
	LabelType       string `json:"labelType"`
	TaskSize        int    `json:"taskSize"`
	SubmitTime      int64  `json:"submitTime"`
	NumSubmissions  int    `json:"numSubmissions"`
	NumLabeledItems int    `json:"numLabeledItems"`
	StartTime       int64  `json:"startTime"`
}

// An event describing a user action.
type Event struct {
	Timestamp   int64       `json:"timestamp"`
	Action      string      `json:"action"`
	TargetIndex string      `json:"targetIndex"`
	Position    interface{} `json:"position"`
}

// An item and associated metadata.
type ItemObject struct {
	Url         string   `json:"url"`
	GroundTruth string   `json:"groundTruth"`
	Labels      []Label  `json:"labels"`
	Tags        []string `json:"tags"`
}

// TODO: remove? Not used.
type Label struct {
	Id        string      `json:"id"`
	Category  string      `json:"category"`
	Attribute interface{} `json:"attribute"`
	Position  interface{} `json:"position"`
}
