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

// DEPRECATED
func videoLabelingHandler(w http.ResponseWriter, r *http.Request) {
	// use template to insert frames location
	tmpl, err := template.ParseFiles(env.VideoPath())
	if err != nil {
		Error.Println(err)
	}
	executeLabelingTemplate(w, r, tmpl)
}
