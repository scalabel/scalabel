package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"html/template"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"
	"strings"
)

func parse(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		if strings.ContainsRune(r.URL.Path, '.') {
			mux.ServeHTTP(w, r)
			return
		}
		h.ServeHTTP(w, r)
	}
}

func indexHandler(w http.ResponseWriter, r *http.Request) {
	w.Write(HTML)
}

func dashboardHandler(w http.ResponseWriter, r *http.Request) {
	// use template to insert assignment links
	tmpl, err := template.ParseFiles(env.ProjectPath + "/app/control/monitor.html")
	if err != nil {
		Error.Println(err)
		http.NotFound(w, r)
		return
	}
	dbContents := DashboardContents{}
	dbContents.Tasks = GetTasks()
	dbContents.VideoTasks = GetVideoTasks()
	fmt.Println(dbContents.VideoTasks)
	tmpl.Execute(w, dbContents)
}

func box2DLabelingHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.ProjectPath + "/app/annotation/box.html")
	if err != nil {
		log.Fatal(err)
	}
	// get task name from the URL
	projName := r.URL.Query()["project_name"][0]
	taskName := r.URL.Query()["task_id"][0]

	task := GetTask(projName, taskName)
	tmpl.Execute(w, task)
}

func postAssignmentHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}

	var task = Task{}
	// Process image list file
	r.ParseMultipartForm(32 << 20)
	file, _, err := r.FormFile("item_list")
	defer file.Close()
	json.NewDecoder(file).Decode(&task)

	// Process label categories file
	labelFile, _, err := r.FormFile("label")
	var labels []string
	scanner := bufio.NewScanner(labelFile)
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}

	// Process attributes file
	// This holds a map of strings to arbitrary data types.
	var customAttributes map[string]interface{}
	attributeFile, _, err := r.FormFile("custom_attributes")
	if err != nil {
		Error.Println("Failed to load the attribute file")
	} else {
		defer attributeFile.Close()
		json.NewDecoder(attributeFile).Decode(&customAttributes)
		Info.Println(customAttributes)
	}

	taskSize, err := strconv.Atoi(r.FormValue("task_size"))
	task.ProjectName = r.FormValue("project_name")

	size := len(task.Items)
	assignmentID := 0
	for i := 0; i < size; i += taskSize {

		// Initialize new assignment
		assignment := Task{
			ProjectName:     r.FormValue("project_name"),
			LabelType:       r.FormValue("label_type"),
			Category:        labels,
			VendorID:        r.FormValue("vendor_id"),
			AssignmentID:    formatID(assignmentID),
			WorkerID:        strconv.Itoa(assignmentID),
			NumLabeledItems: 0,
			NumSubmissions:  0,
			StartTime:       recordTimestamp(),
			Items:           task.Items[i:Min(i+taskSize, size)],
			TaskSize:        taskSize,
		}

		assignmentID = assignmentID + 1

		// Save assignment to data folder
		assignmentPath := assignment.GetAssignmentPath()

		assignmentJson, _ := json.MarshalIndent(assignment, "", "  ")
		err = ioutil.WriteFile(assignmentPath, assignmentJson, 0644)

		if err != nil {
			Error.Println("Failed to save assignment file of",
				assignment.ProjectName, assignment.AssignmentID)
		} else {
			Info.Println("Saving assignment file of",
				assignment.ProjectName, assignment.AssignmentID)
		}
	}

	Info.Println("Created", assignmentID, "new assignments")

	w.Write([]byte(strconv.Itoa(assignmentID)))
}

func postSubmissionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		Error.Println("Failed to read submission request body")
	}
	assignment := Task{}
	err = json.Unmarshal(body, &assignment)
	if err != nil {
		Error.Println("Failed to parse submission JSON")
	}

	if assignment.NumLabeledItems == assignment.TaskSize {
		assignment.NumSubmissions = assignment.NumSubmissions + 1
		Info.Println("Complete submission of",
			assignment.ProjectName, assignment.AssignmentID)
	}
	assignment.SubmitTime = recordTimestamp()

	submissionPath := assignment.GetSubmissionPath()
	taskJson, _ := json.MarshalIndent(assignment, "", "  ")

	err = ioutil.WriteFile(submissionPath, taskJson, 0644)
	if err != nil {
		Error.Println("Failed to save submission file of",
			assignment.ProjectName, assignment.AssignmentID,
			"for Path:", submissionPath)
	}

	latestSubmissionPath := assignment.GetLatestSubmissionPath()
	latestTaskJson, _ := json.MarshalIndent(assignment, "", "  ")
	err = ioutil.WriteFile(latestSubmissionPath, latestTaskJson, 0644)
	if err != nil {
		Error.Println("Failed to save latest submission file of",
			assignment.ProjectName, assignment.AssignmentID)
	}
	// Debug
	w.Write(taskJson)

}

func postLogHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		Error.Println("Failed to read log request body")
	}
	assignment := Task{}
	err = json.Unmarshal(body, &assignment)
	if err != nil {
		Error.Println("Failed to parse log JSON")
	}

	if assignment.NumLabeledItems == assignment.TaskSize {
		assignment.NumSubmissions = assignment.NumSubmissions + 1
	}

	assignment.SubmitTime = recordTimestamp()
	// Save to Log every five images displayed
	logPath := assignment.GetLogPath()
	taskJson, _ := json.MarshalIndent(assignment, "", "  ")
	err = ioutil.WriteFile(logPath, taskJson, 0644)
	if err != nil {
		Error.Println("Failed to save log file of",
			assignment.ProjectName, assignment.AssignmentID)
	} else {
		Info.Println("Saving log of",
			assignment.ProjectName, assignment.AssignmentID)
	}

	w.Write(taskJson)
}

func requestAssignmentHandler(w http.ResponseWriter, r *http.Request) {

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		Error.Println("Failed to read assignment request body")
	}
	task := Task{}
	err = json.Unmarshal(body, &task)
	if err != nil {
		Error.Println("Failed to parse assignment request JSON")
	}
	requestPath := task.GetAssignmentPath()

	requestJson, err := ioutil.ReadFile(requestPath)
	if err != nil {
		Error.Println("Failed to read assignment file of",
			task.ProjectName, task.AssignmentID)
	} else {
		Info.Println("Finished reading assignment file of",
			task.ProjectName, task.AssignmentID)
	}
	w.Write(requestJson)

}

func requestSubmissionHandler(w http.ResponseWriter, r *http.Request) {

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		Error.Println("Failed to read submission request body")
	}
	request := Task{}
	err = json.Unmarshal(body, &request)
	if err != nil {
		Error.Println("Failed to parse submission request JSON")
	}
	requestPath := request.GetLatestSubmissionPath()
	assignmentPath := request.GetAssignmentPath()

	var existingPath string
	if Exists(requestPath) {
		existingPath = requestPath
	} else if Exists(assignmentPath) {
		existingPath = assignmentPath
	} else {
		Error.Println("Can not find",
			request.ProjectName, request.AssignmentID)
		http.NotFound(w, r)
		return
	}

	requestJson, err := ioutil.ReadFile(existingPath)
	if err != nil {
		Error.Println("Failed to read submission file of",
			request.ProjectName, request.AssignmentID)
	} else {
		Info.Println("Loading assignment from latest submission of",
			request.ProjectName, request.AssignmentID)
	}

	w.Write(requestJson)

}

func requestInfoHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		Error.Println("Failed to read submission request body")
	}
	request := Task{}
	err = json.Unmarshal(body, &request)
	if err != nil {
		Error.Println("Failed to parse submission request JSON")
	}
	requestPath := request.GetLatestSubmissionPath()
	assignmentPath := request.GetAssignmentPath()

	var existingPath string
	if Exists(requestPath) {
		existingPath = requestPath
	} else if Exists(assignmentPath) {
		existingPath = assignmentPath
	} else {
		Error.Println("Can not find", assignmentPath,
			request.ProjectName, request.AssignmentID)
		http.NotFound(w, r)
		return
	}

	requestJson, err := ioutil.ReadFile(existingPath)
	if err != nil {
		Error.Println("Failed to read submission file of",
			request.ProjectName, request.AssignmentID)
	} else {
		Info.Println("Loading task info of",
			request.ProjectName, request.AssignmentID)
	}

	info := TaskInfo{}
	json.Unmarshal(requestJson, &info)

	infoJson, _ := json.MarshalIndent(info, "", "  ")
	w.Write(infoJson)

}

func readResultHandler(w http.ResponseWriter, r *http.Request) {
	queryValues := r.URL.Query()
	filename := queryValues.Get("task_id")
	projectName := queryValues.Get("project_name")

	HTML = GetResult(filename, projectName)
	w.Write(HTML)
}

func readFullResultHandler(w http.ResponseWriter, r *http.Request) {
	queryValues := r.URL.Query()
	projectName := queryValues.Get("project_name")

	HTML = GetFullResult(projectName)
	w.Write(HTML)
}
