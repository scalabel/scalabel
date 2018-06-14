package main

import (
	"html/template"
	"net/http"
)

func box2dLabelingHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.Box2dPath())
	if err != nil {
		Error.Println(err)
	}
	// get task name from the URL
	projectName := r.URL.Query()["project_name"][0]
	taskIndex := r.URL.Query()["task_index"][0]

	assignment := GetAssignment(projectName, taskIndex)
	Info.Println(assignment)
	tmpl.Execute(w, assignment)
}

func seg2dLabelingHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.Seg2dPath())
	if err != nil {
		Error.Println(err)
	}
	// get task name from the URL
	projectName := r.URL.Query()["project_name"][0]
	taskIndex := r.URL.Query()["task_index"][0]

	assignment := GetAssignment(projectName, taskIndex)
	Info.Println(assignment)
	tmpl.Execute(w, assignment)
}

func lane2dLabelingHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.Lane2dPath())
	if err != nil {
		Error.Println(err)
	}
	// get task name from the URL
	projectName := r.URL.Query()["project_name"][0]
	taskIndex := r.URL.Query()["task_index"][0]

	assignment := GetAssignment(projectName, taskIndex)
	Info.Println(assignment)
	tmpl.Execute(w, assignment)
}
