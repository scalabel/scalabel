package main

import (
	"bufio"
	"encoding/json"
	"html/template"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"
	"strconv"
)

// A video annotation task to be completed by a user.
type VideoTask struct {
	ProjectName string        `json:"projectName"`
	WorkerID    string        `json:workerId"`
	Category    []string      `json:"category"`
	LabelType   string        `json:"labelType"`
	TaskName    string        `json:"taskName"`
	TaskURL     string        `json:"taskURL"`
	VideoName   string        `json:"videoName"`
	Metadata    VideoMetadata `json:"metadata"`
	SubmitTime  int64         `json:"submitTime"`
	StartTime   int64         `json:"startTime"`
	Events      []Event       `json:"events"`
	VendorID    string        `json:"vendorId"`
	IPAddress   interface{}   `json:"ipAddress"`
	UserAgent   string        `json:"userAgent"`
}

// TODO: move "Assignments" to "Tasks"
func (task *VideoTask) GetVideoTaskPath() string {
	dir := path.Join(env.ProjectPath,
		"data",
		"Assignments",
		"video",
		task.ProjectName,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, task.TaskName+".json")
}

func (task *VideoTask) GetVideoSubmissionPath() string {
	dir := path.Join(env.ProjectPath,
		"data",
		"Submissions",
		"video",
		task.ProjectName,
		task.WorkerID,
	)
	os.MkdirAll(dir, 0777)
	startTime := formatTime(task.StartTime)
	return path.Join(dir, startTime+".json")
}

func (task *VideoTask) GetLatestVideoSubmissionPath() string {
	dir := path.Join(env.ProjectPath,
		"data",
		"Submissions",
		"video",
		task.ProjectName,
		task.WorkerID,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, "latest.json")
}

func (task *VideoTask) GetVideoLogPath() string {
	dir := path.Join(env.ProjectPath,
		"Log",
		"video",
		task.ProjectName,
		task.WorkerID,
	)
	os.MkdirAll(dir, 0777)
	submitTime := formatTime(task.SubmitTime)
	return path.Join(dir, submitTime+".json")
}

// Metadata describing a video
type VideoMetadata struct {
	Bitrate    string `json:"bitrate"`
	TBR        string `json:"tbr"`
	FPS        string `json:"fps"`
	TBN        string `json:"tbn"`
	TBC        string `json:"tbc"`
	NumFrames  string `json:"numFrames"`
	Resolution string `json:"resolution"`
}

func videoLabelingHandler(w http.ResponseWriter, r *http.Request) {
	// use template to insert frames location
	tmpl, err := template.ParseFiles(env.ProjectPath + "/app/annotation/video.html")
	if err != nil {
		log.Fatal(err)
	}
	// get task name from the URL
	// TODO: handle missing task name
	projName := r.URL.Query()["project_name"][0]
	taskName := r.URL.Query()["task_name"][0]

	task := GetVideoTask(projName, taskName)
	tmpl.Execute(w, task)
}

func postVideoAssignmentHandler(w http.ResponseWriter, r *http.Request) {
	// don't want non-POST http method
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}

	// read the video name
	videoName := r.FormValue("video_name")
	// get the path of the vid and vid frames
	videoPath := env.DataDir + "/videos/" + videoName
	framePath := env.DataDir + "/frames/" + videoName
	framePath = framePath[:len(framePath)-4] // take off the .mp4

	// if no frames directory for this vid, throw error
	_, err := os.Stat(framePath)
	if err != nil {
		Error.Println(videoPath + " has not been split into frames.")
		http.NotFound(w, r)
		return
	}

	// read the metadata file
	// TODO: error handling
	mdContents, _ := ioutil.ReadFile(framePath + "/metadata.json")
	vmd := VideoMetadata{}
	json.Unmarshal(mdContents, &vmd)

	// read the object categories file
	objCatsFile, _, err := r.FormFile("object_categories")
	var objCats []string
	scanner := bufio.NewScanner(objCatsFile)
	for scanner.Scan() {
		objCats = append(objCats, scanner.Text())
	}

	projectName := r.FormValue("project_name")
	taskName := r.FormValue("task_name")
	task := VideoTask{
		ProjectName: projectName,
		// WorkerID:    -1, // TODO
		Category:  objCats,
		LabelType: "2d_bbox",
		TaskName:  taskName,
		TaskURL:   "/video_bbox_labeling?project_name=" + projectName + "&task_name=" + taskName,
		VideoName: videoName,
		Metadata:  vmd,
		// SubmitTime:  -1, // TODO
		// StartTime:   -1, // TODO
		// Events:      [], // TODO
		// VendorID:    -1, // TODO
		// IPAddress:   -1, // TODO
		// UserAgent:   "", // TODO
	}

	// Save assignment to data folder
	taskPath := task.GetVideoTaskPath()

	taskJson, _ := json.MarshalIndent(task, "", "  ")
	err = ioutil.WriteFile(taskPath, taskJson, 0644)

	if err != nil {
		Error.Println("Failed to save video task file:",
			task.ProjectName, task.TaskName)
	} else {
		Info.Println("Saving video task file:",
			task.ProjectName, task.TaskName)
	}

	Info.Println("Created new video task")

	w.Write([]byte(strconv.Itoa(1))) // TODO - something meaningful here?
}

// TODO
func postVideoSubmissionHandler(w http.ResponseWriter, r *http.Request) {

}

// TODO
func postVideoLogHandler(w http.ResponseWriter, r *http.Request) {

}

// TODO
func requestVideoTaskHandler(w http.ResponseWriter, r *http.Request) {

}

// TODO
func requestVideoSubmissionHandler(w http.ResponseWriter, r *http.Request) {

}

// TODO
func requestVideoInfoHandler(w http.ResponseWriter, r *http.Request) {

}

// TODO
func readVideoResultHandler(w http.ResponseWriter, r *http.Request) {

}

// TODO
func readFullVideoResultHandler(w http.ResponseWriter, r *http.Request) {

}
