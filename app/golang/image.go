package main

import (
	"html/template"
	"net/http"
)

func box2dLabelingHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.ProjectPath + "/app/annotation/box.html")
	if err != nil {
		Error.Println(err)
	}
	// get task name from the URL
	projectName := r.URL.Query()["project_name"][0]
	taskIndex := r.URL.Query()["task_index"][0]

	task := GetTask(projectName, taskIndex)
	tmpl.Execute(w, task)
}
