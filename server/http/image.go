package main

import (
	"html/template"
	"net/http"
)

func Label2dHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.Label2dPath())
	if err != nil {
		Error.Println(err)
	}
	executeLabelingTemplate(w, r, tmpl)
}

func Label3dHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.Label3dPath())
	if err != nil {
		Error.Println(err)
	}
	executeLabelingTemplate(w, r, tmpl)
}

func pointCloudTrackingHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.PointCloudTrackingPath())
	if err != nil {
		Error.Println(err)
	}
	executeLabelingTemplate(w, r, tmpl)
}
