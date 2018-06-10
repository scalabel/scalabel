package sat

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

	task := GetTask(projectName, taskIndex)
	Info.Println(task)
	tmpl.Execute(w, task)
}
