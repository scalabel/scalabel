package main

import (
	"html/template"
	"net/http"
)

// Metadata describing a video
type VideoMetaData struct {
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
	tmpl, err := template.ParseFiles(env.VideoPath())
	if err != nil {
		Error.Println(err)
	}
	// get task index from the URL
	projectName := r.URL.Query()["project_name"][0]
	taskIndex := r.URL.Query()["task_index"][0]

	assignment := GetAssignment(projectName, taskIndex)
	Info.Println(assignment)
	tmpl.Execute(w, assignment)
}
