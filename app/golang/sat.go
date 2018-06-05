package main

import (
	"bytes"
	"encoding/json"
	"gopkg.in/yaml.v2"
	"html/template"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"strconv"
	"strings"
)

// Project is what the admin creates, specifying a list of items
type Project struct {
	Name          string        `json:"name" yaml:"name"`
	ItemType      string        `json:"itemType" yaml:"itemType"`
	LabelType     string        `json:"labelType" yaml:"labelType"`
	Items         []Item        `json:"items" yaml"items"`
	Categories    []Category    `json:"categories" yaml:"categories"`
	TaskSize      int           `json:"taskSize" yaml:"taskSize"`
	Attributes    []Attribute   `json:"attributes" yaml:"attributes"`
	VendorId      int           `json:"vendorId" yaml:"vendorId"`
	VideoMetadata VideoMetadata `json:"metadata" yaml:"metadata"`
}

// A chunk of a project
type Task struct {
	HandlerUrl   string         `json:"handlerUrl" yaml:"handlerUrl"`
	ProjectName  string         `json:"projectName" yaml:"projectName"`
	Index        int            `json:"index" yaml:"index"`
	Items        []Item         `json:"items" yaml:"items"`
	Labels       []Label        `json:"labels" yaml:"labels"`
	Categories   []Category     `json:"categories" yaml:"categories"`
	Attributes   []Attribute    `json:"attributes" yaml:"attributes"`
	VideoMetadata VideoMetadata `json:"metadata" yaml:"metadata"`
}

// The actual assignment of a task to an annotator
type Assignment struct {
	Task      Task              `json:"task" yaml:"task"`
	WorkerId  int               `json:"workerId" yaml:"workerId"`
	Events    []Event           `json:"events" yaml:"events"`
	UserAgent string            `json:"userAgent" yaml:"userAgent"`
	IpInfo    map[string]string `json:ipInfo" yaml:"ipInfo"`
}

// An item is something to be annotated e.g. Image, PointCloud
type Item struct {
	Url         string  `json:"url" yaml:"url"`
	Index       int     `json:"index" yaml:"index"`
	LabelIds    []int   `json:"labelIds" yaml:"labelIds"`
	GroundTruth []Label `json:"groundTruth" yaml:"groundTruth"`
}

// An annotation for an item, needs to include all possible annotation types
type Label struct {
	Id              int                `json:"id" yaml:"id"`
	CategoryPath    string             `json:"categoryPath" yaml:"categoryPath"`
	ParentId        int                `json:"parent" yaml:"parentId"`
	ChildrenIds     []int              `json:"children" yaml:"childrenIds"`
	PreviousLabelId int                `json:"previousLabelId" yaml:"previousLabelId"`
	NextLabelId     int                `json:"nextLabelId" yaml:"nextLabelId"`
	AttributeValues map[string]bool    `json:"attributeValues" yaml:"attributeValues"`
	Box2d           map[string]float32 `json:"box2d" yaml:"box2d"`
	Keyframe        bool               `json:"keyframe" yaml:"keyframe"`
}

// A class value for a label.
type Category struct {
	Name          string     `json:"name" yaml:"name"`
	Subcategories []Category `json:"subcategories" yaml:"subcategories"`
}

// A configurable attribute describing a label
type Attribute struct {
	Name         string   `json:"name" yaml:"name"`
	ToolType     string   `json:"toolType" yaml:"toolType"`
	TagText      string   `json:"tagText" yaml:"tagText"`
	TagPrefix    string   `json:"tagPrefix" yaml:"tagPrefix"`
	TagSuffixes  []string `json:"tagSuffixes" yaml:"tagSuffixes"`
	Values       []string `json:"values" yaml:"values"`
	ButtonColors []string `json:"buttonColors" yaml:"buttonColors"`
}

// An event describing an annotator's interaction with the session
type Event struct {
}

func parse(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if strings.ContainsRune(r.URL.Path, '.') {
			mux.ServeHTTP(w, r)
			return
		}
		h.ServeHTTP(w, r)
	}
}

// Function type for handlers
type handler func(http.ResponseWriter, *http.Request)

// MakeStandardHandler returns a function for handling static HTML
func MakeStandardHandler(pagePath string) handler {
	return func(w http.ResponseWriter, r *http.Request) {
		HTML, err := ioutil.ReadFile(env.ProjectPath + pagePath)
		if err != nil {
			Error.Println(err)
		}
		w.Write(HTML)
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
	Info.Println(GetTasks())
	tmpl.Execute(w, GetTasks())
}

// TODO: split this function up
// Handles the posting of new projects
func postProjectHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}

	// item list YAML
	itemType := r.FormValue("item_type")
	var items []Item
	itemFile, _, err := r.FormFile("item_file")
	defer itemFile.Close()
	if err != nil {
		Error.Println(err)
	}
	itemFileBuf := bytes.NewBuffer(nil)
	_, err = io.Copy(itemFileBuf, itemFile)
	if err != nil {
		Error.Println(err)
	}
	var vmd VideoMetadata
	if itemType == "video" {
		var frameDirectoryItems []Item
		err = yaml.Unmarshal(itemFileBuf.Bytes(), &frameDirectoryItems)
		if err != nil {
			Error.Println(err)
		}
		// in video, we only consider the first item, and
		//   the item url is the frame directory
		frameUrl := frameDirectoryItems[0].Url
		framePath := env.DataDir + frameUrl[1:len(frameUrl)]
		// if no frames directory for this vid, throw error
		_, err := os.Stat(framePath)
		if err != nil {
			Error.Println(framePath + " does not exist. Has video been split into frames?")
			http.NotFound(w, r)
			return
		}
		// get the video's metadata
		mdContents, _ := ioutil.ReadFile(framePath + "/metadata.json")
		json.Unmarshal(mdContents, &vmd)
		// get the URLs of all frame images
		numFrames, err := strconv.Atoi(vmd.NumFrames)
		if err != nil {
			Error.Println(err)
		}
		for i := 0; i < numFrames; i++ {
			frameString := strconv.Itoa(i + 1)
			for len(frameString) < 7 {
				frameString = "0" + frameString
			}
			frameItem := Item{
				Url:   frameUrl + "/f-" + frameString + ".jpg",
				Index: i,
			}
			items = append(items, frameItem)
		}
	} else {
		err = yaml.Unmarshal(itemFileBuf.Bytes(), &items)
		if err != nil {
			Error.Println(err)
		}
		for i := 0; i < len(items); i++ {
			items[i].Index = i;
		}
	}

	// categories YAML
	var categories []Category
	categoryFile, _, err := r.FormFile("categories")
	defer categoryFile.Close()
	if err != nil {
		Error.Println(err)
	}
	categoryFileBuf := bytes.NewBuffer(nil)
	_, err = io.Copy(categoryFileBuf, categoryFile)
	if err != nil {
		Error.Println(err)
	}
	err = yaml.Unmarshal(categoryFileBuf.Bytes(), &categories)
	if err != nil {
		Error.Println(err)
	}

	// attributes YAML
	var attributes []Attribute
	attributeFile, _, err := r.FormFile("attributes")
	defer attributeFile.Close()
	if err != nil {
		Error.Println(err)
	}
	attributeFileBuf := bytes.NewBuffer(nil)
	_, err = io.Copy(attributeFileBuf, attributeFile)
	if err != nil {
		Error.Println(err)
	}
	err = yaml.Unmarshal(attributeFileBuf.Bytes(), &attributes)
	if err != nil {
		Error.Println(err)
	}

	// parse the task size
	taskSize, err := strconv.Atoi(r.FormValue("task_size"))
	if err != nil {
		Error.Println(err)
	}
	vendorId, err := strconv.Atoi(r.FormValue("vendor_id"))
	if err != nil {
		Error.Println(err)
	}
	var project = Project{
		Name:          r.FormValue("project_name"),
		ItemType:      r.FormValue("item_type"),
		LabelType:     r.FormValue("label_type"),
		Items:         items,
		Categories:    categories,
		TaskSize:      taskSize,
		Attributes:    attributes,
		VendorId:      vendorId,
		VideoMetadata: vmd,
	}

	index := 0
	handlerUrl := GetHandlerUrl(project)
	if itemType == "video" {
		task := Task{
			HandlerUrl:    handlerUrl,
			ProjectName:   project.Name,
			Index:         0,
			Items:         project.Items,
			Categories:    project.Categories,
			Attributes:    project.Attributes,
			VideoMetadata: vmd,
		}
		index = 1

		// Save task to task folder
		taskPath := task.GetTaskPath()
		taskJson, _ := json.MarshalIndent(task, "", "  ")
		err = ioutil.WriteFile(taskPath, taskJson, 0644)
		if err != nil {
			Error.Println("Failed to save task file of", task.ProjectName,
				task.Index)
		} else {
			Info.Println("Saving task file of", task.ProjectName,
				task.Index)
		}
	} else {
		size := len(project.Items)
		for i := 0; i < size; i += taskSize {
			// Initialize new task
			task := Task{
				HandlerUrl:  handlerUrl,
				ProjectName: project.Name,
				Index:       index,
				Items:       project.Items[i:Min(i+taskSize, size)],
				Categories:  project.Categories,
				Attributes:  project.Attributes,
			}
			index = index + 1

			// Save task to task folder
			taskPath := task.GetTaskPath()
			taskJson, _ := json.MarshalIndent(task, "", "  ")
			err = ioutil.WriteFile(taskPath, taskJson, 0644)

			if err != nil {
				Error.Println("Failed to save task file of", task.ProjectName,
					task.Index)
			} else {
				Info.Println("Saving task file of", task.ProjectName,
					task.Index)
			}
		}
	}

	Info.Println("Created", index, "new tasks")

	// TODO: is this necessary?
	w.Write([]byte(strconv.Itoa(index)))
}

// Return all of the tasks.
func postGetTasksHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}

	tasks := GetTasks()
	tasksJson, err := json.Marshal(tasks)
	if err != nil {
		Error.Println(err)
	}
	w.Write(tasksJson)
}

func postLoadTaskHandler(w http.ResponseWriter, r *http.Request) {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		Error.Println(err)
	}
	taskToLoad := Task{}
	err = json.Unmarshal(body, &taskToLoad)
	if err != nil {
		Error.Println(err)
	}
	loadedTask := GetTask(taskToLoad.ProjectName, strconv.Itoa(taskToLoad.Index))
	loadedTaskJson, err := json.Marshal(loadedTask)
	if err != nil {
		Error.Println(err)
	}
	w.Write(loadedTaskJson)
}

// Handles the posting of saved tasks
func postSaveHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		Error.Println(err)
	}
	task := Task{}
	err = json.Unmarshal(body, &task)
	if err != nil {
		Error.Println(err)
	}
	Info.Println(task)

	taskPath := path.Join(env.DataDir, "Tasks", task.ProjectName, strconv.Itoa(task.Index) + ".json")
	taskJson, err := json.MarshalIndent(task, "", "  ")
	if err != nil {
		Error.Println(err)
	}

	err = ioutil.WriteFile(taskPath, taskJson, 0644)
	if err != nil {
		Error.Println(err)
	} else {
		Info.Println("Saved task " + taskPath)
	}

	w.Write(nil)
}

// Handles the posting of completed tasks
func postSubmissionHandler(w http.ResponseWriter, r *http.Request) {
	// TODO
}
