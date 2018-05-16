package main

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"
	"strconv"
	"time"
	"unicode/utf8"
)

// Function type for handlers
type handler func(http.ResponseWriter, *http.Request)

// serveStaticDirectory serves a directory in the project located within the
// parentFolder as that same directory name under the root of the web
// directory.
// TODO
func serveStaticDirectory(parentFolder string, dir string) {
	fileServer := http.FileServer(http.Dir(env.ProjectPath + "/" + parentFolder + "/" + dir))
	strippedHandler := http.StripPrefix("/"+dir+"/", fileServer)
	http.Handle("/"+dir+"/", strippedHandler)
}

// MakeStandardHandler returns a function for handling static HTML
func MakeStandardHandler(pagePath string) handler {
	return func(w http.ResponseWriter, r *http.Request) {
		HTML, err := ioutil.ReadFile(env.ProjectPath + pagePath)
		if err != nil {
			log.Fatal(err) // TODO: send stress signal
		}
		w.Write(HTML)
	}
}

func GetTask(projName string, taskName string) Task {
	// TODO: account for projType directory structure
	taskPath := path.Join(env.ProjectPath,
		"data",
		"Assignments",
		projName,
		taskName+".json",
	)

	// TODO: error handling
	fileContents, _ := ioutil.ReadFile(taskPath)
	task := Task{}
	json.Unmarshal(fileContents, &task)
	return task
}

// TODO
func GetTasks() []Task {
	dirPath := path.Join(env.ProjectPath,
		"data",
		"Assignments",
		"img",
	)
	// TODO: error handling
	dirContents, _ := ioutil.ReadDir(dirPath)
	tasks := []Task{}
	for _, file := range dirContents {
		fileContents, _ := ioutil.ReadFile(dirPath + file.Name())
		t := Task{}
		json.Unmarshal(fileContents, &t)
		tasks = append(tasks, t)
	}
	return tasks
}

// TODO
func GetVideoTask(projName string, taskName string) VideoTask {
	// TODO: move "Assignments" to "Tasks"
	projPath := path.Join(env.ProjectPath,
		"data",
		"Assignments",
		"video",
		projName,
	)
	// TODO: error handling
	fileContents, _ := ioutil.ReadFile(path.Join(projPath, taskName+".json"))
	task := VideoTask{}
	json.Unmarshal(fileContents, &task)
	return task
}

// TODO
func GetVideoTasks() []VideoTask {
	dirPath := path.Join(env.ProjectPath,
		"data",
		"Assignments",
		"video",
	)
	// TODO: error handling
	dirContents, _ := ioutil.ReadDir(dirPath)
	tasks := []VideoTask{}
	// loop through all the projects
	for _, proj := range dirContents {
		if !proj.IsDir() {
			continue
		}
		projContents, _ := ioutil.ReadDir(path.Join(dirPath, proj.Name()))
		for _, file := range projContents {
			if file.Name()[0:1] == "." {
				continue
			}
			fileContents, _ := ioutil.ReadFile(path.Join(dirPath, proj.Name(), file.Name()))
			t := VideoTask{}
			json.Unmarshal(fileContents, &t)
			tasks = append(tasks, t)
		}
	}
	return tasks
}

func (assignment *Task) GetAssignmentPath() string {
	filename := assignment.AssignmentID
	dir := path.Join(
		env.DataDir,
		"Assignments",
		assignment.ProjectName,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, filename+".json")
}

func (assignment *Task) GetSubmissionPath() string {
	startTime := formatTime(assignment.StartTime)
	dir := path.Join(
		env.DataDir,
		"Submissions",
		assignment.ProjectName,
		assignment.AssignmentID,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, startTime+".json")
}

func (assignment *Task) GetLatestSubmissionPath() string {
	dir := path.Join(
		env.DataDir,
		"Submissions",
		assignment.ProjectName,
		assignment.AssignmentID,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, "latest.json")
}

func (assignment *Task) GetLogPath() string {
	submitTime := formatTime(assignment.SubmitTime)
	dir := path.Join(
		env.DataDir,
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
	submissionPath := path.Join(
		env.DataDir,
		"Submissions",
		projectName,
		assignmentID,
		"latest.json",
	)
	assignmentPath := path.Join(
		env.DataDir,
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
		result.Items = task.Items
	}
	resultJson, _ := json.MarshalIndent(result, "", "  ")

	return resultJson
}

func GetFullResult(projectName string) []byte {
	result := Result{}
	dir := path.Join(
		env.DataDir,
		"Submissions",
		projectName,
	)
	files, _ := ioutil.ReadDir(dir)
	for _, f := range files {
		filename := f.Name()
		submissionPath := path.Join(
			dir,
			filename,
			"latest.json",
		)
		assignmentPath := path.Join(
			env.DataDir,
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
			for i := 0; i < len(task.Items); i += 1 {
				result.Items = append(result.Items, task.Items[i])
			}
		}

	}
	fullResultJson, _ := json.MarshalIndent(result, "", "  ")

	return fullResultJson
}
