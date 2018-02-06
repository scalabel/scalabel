package main

import (
	"encoding/json"
	"go/build"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"strconv"
	"time"
	"unicode/utf8"
)

// GetProjPath returns the path of this go project. It assumes setup of the go
// environment according to: https://golang.org/doc/code.html#Workspaces
func GetProjPath() string {
	gopath := os.Getenv("GOPATH")
	if gopath == "" {
        gopath = build.Default.GOPATH
    }
	return gopath + "/src/sat" // TODO: move project name to config
}

// Function type for handlers
type handler func(http.ResponseWriter, *http.Request)

// MakeStandardHandler returns a function for handling static HTML
func MakeStandardHandler(pagePath string) (handler) {
	return func(w http.ResponseWriter, r *http.Request) {
		HTML, _ = ioutil.ReadFile(GetProjPath() + pagePath)
		w.Write(HTML)
	}
}

func (assignment *Task) GetAssignmentPath() string {
	filename := assignment.AssignmentID
	dir := path.Join(GetProjPath(),
		"data",
		"Assignments",
		assignment.ProjectName,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, filename+".json")
}

func (assignment *Task) GetSubmissionPath() string {
	startTime := formatTime(assignment.StartTime)
	dir := path.Join(GetProjPath(),
		"data",
		"Submissions",
		assignment.ProjectName,
		assignment.AssignmentID,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, startTime+".json")
}

func (assignment *Task) GetLatestSubmissionPath() string {
	dir := path.Join(GetProjPath(),
		"data",
		"Submissions",
		assignment.ProjectName,
		assignment.AssignmentID,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, "latest.json")
}

func (assignment *Task) GetLogPath() string {
	submitTime := formatTime(assignment.SubmitTime)
	dir := path.Join(GetProjPath(),
		"Log",
		assignment.ProjectName,
		assignment.AssignmentID,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, submitTime+".json")
}

func recordTimestamp() int64 {
	// record timestamp in seconds
	return time.Now().Unix()
}

func formatTime(timestamp int64) string {
	t := time.Unix(timestamp, 0)
	return t.Format("2006-01-02_03-04-05")
}

func formatID(id int) string {
	str := strconv.Itoa(id)
	strLen := utf8.RuneCountInString(str)
	for i := 0; i < (4 - strLen); i += 1 {
		str = "0" + str
	}
	return str
}

func Exists(name string) bool {
	_, err := os.Stat(name)
	if os.IsNotExist(err) {
		return false
	} else {
		return true
	}
}

func Min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func GetResult(assignmentID string, projectName string) []byte {
	submissionPath := path.Join(GetProjPath(),
		"Submissions",
		projectName,
		assignmentID,
		"latest.json",
	)
	assignmentPath := path.Join(GetProjPath(),
		"Assignments",
		projectName,
		assignmentID+".json",
	)

	result := Result{}

	var existingPath string
	if Exists(submissionPath) {
		existingPath = submissionPath
	} else if Exists(assignmentPath) {
		existingPath = assignmentPath
	}

	if len(existingPath) > 0 {
		taskJson, err := ioutil.ReadFile(existingPath)
		if err != nil {
			Error.Println("Failed to read result of",
				projectName, assignmentID)
		} else {
			Info.Println("Reading result of",
				projectName, assignmentID)
		}

		task := Task{}
		json.Unmarshal(taskJson, &task)
		result.Images = task.Images
	}
	resultJson, _ := json.MarshalIndent(result, "", "  ")

	return resultJson
}

func GetFullResult(projectName string) []byte {
	result := Result{}
	dir := path.Join(GetProjPath(),
		"Submissions",
		projectName,
	)
	files, _ := ioutil.ReadDir(dir)
	for _, f := range files {
		filename := f.Name()
		submissionPath := path.Join(GetProjPath(),
			filename,
			"latest.json",
		)
		assignmentPath := path.Join(GetProjPath(),
			"Assignments",
			projectName,
			filename+".json",
		)

		var existingPath string
		if Exists(submissionPath) {
			existingPath = submissionPath
		} else if Exists(assignmentPath) {
			existingPath = assignmentPath
		}

		if len(existingPath) > 0 {

			resultJson, err := ioutil.ReadFile(existingPath)
			if err != nil {
				Error.Println("Failed to read result of", projectName)
			} else {
				Info.Println("Reading result of", projectName)
			}

			task := Task{}
			json.Unmarshal(resultJson, &task)
			for i := 0; i < len(task.Images); i += 1 {
				result.Images = append(result.Images, task.Images[i])
			}
		}

	}
	fullResultJson, _ := json.MarshalIndent(result, "", "  ")

	return fullResultJson
}
