package main

import (
	"html/template"
	"net/http"
)

// TODO: consolidate all labeling handlers
func box2dLabelingHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.Box2dPath())
	Info.Println(env.Box2dPath())
	if err != nil {
		Error.Println(err)
	}
	executeLabelingTemplate(w, r, tmpl)
}

// DEPRECATED
func seg2dLabelingHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.Seg2dPath())
	if err != nil {
		Error.Println(err)
	}
	executeLabelingTemplate(w, r, tmpl)
}

// DEPRECATED
func lane2dLabelingHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.Lane2dPath())
	if err != nil {
		Error.Println(err)
	}
	executeLabelingTemplate(w, r, tmpl)
}

func pointCloudLabelingHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.PointCloudPath())
	if err != nil {
		Error.Println(err)
	}
	executeLabelingTemplate(w, r, tmpl)
}
