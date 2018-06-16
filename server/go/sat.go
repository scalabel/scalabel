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
	"log"
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
	VideoMetaData VideoMetaData `json:"metadata" yaml:"metadata"`
}

// A chunk of a project
type Task struct {
	HandlerUrl    string        `json:"handlerUrl" yaml:"handlerUrl"`
	PageTitle     string        `json:"pageTitle" yaml:"pageTitle"`
	ProjectName   string        `json:"projectName" yaml:"projectName"`
	Index         int           `json:"index" yaml:"index"`
	Items         []Item        `json:"items" yaml:"items"`
	Tracks        []Label       `json:"tracks" yaml:"tracks"`
	Categories    []Category    `json:"categories" yaml:"categories"`
	Attributes    []Attribute   `json:"attributes" yaml:"attributes"`
	VideoMetaData VideoMetaData `json:"metadata" yaml:"metadata"`
}

// The actual assignment of a task to an annotator
type Assignment struct {
	Task     Task           `json:"task" yaml:"task"`
	Labels   []Label        `json:"labels" yaml:"labels"`
	Tracks   []Label        `json:"tracks" yaml:"tracks"`
	WorkerId int            `json:"workerId" yaml:"workerId"`
	Events   []Event        `json:"events" yaml:"events"`
	Info     AssignmentInfo `json:"info" yaml:"info"`
}

// Info describing an assignment
type AssignmentInfo struct {
	SubmitTime        int               `json:"submitTime" yaml:"submitTime"`
	SubmissionsCount  int               `json:"submissionsCount" yaml:"submissionsCount"`
	LabeledItemsCount int               `json:"labeledItemsCount" yaml:"labeledItemsCount"`
	UserAgent         string            `json:"userAgent" yaml:"userAgent"`
	IpInfo            map[string]string `json:"ipInfo" yaml:"ipInfo"`
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
	Id              int                    `json:"id" yaml:"id"`
	CategoryPath    string                 `json:"categoryPath" yaml:"categoryPath"`
	ParentId        int                    `json:"parent" yaml:"parentId"`
	ChildrenIds     []int                  `json:"children" yaml:"childrenIds"`
	Attributes      map[string]interface{} `json:"attributes" yaml:"attributes"`
	Data            map[string]interface{} `json:"data" yaml:"data"`
	Keyframe        bool                   `json:"keyframe" yaml:"keyframe"`
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
	Timestamp  int64     `json:"timestamp" yaml:"timestamp"`
	Action     string    `json:"action" yaml:"action"`
	ItemIndex  int32     `json:"itemIndex" yaml:"itemIndex"`
	LabelIndex int32     `json:"labelIndex" yaml:"labelIndex"`
	Position   []float32 `json:"position" yaml:"position"`
}

// Contains all the info needed in the dashboards
type DashboardContents struct {
	Project     Project      `json:"project" yaml:"project"`
	Assignments []Assignment `json:"assignment" yaml:"assignment"`
}

// Function type for handlers
type HandleFunc func(http.ResponseWriter, *http.Request)

// MakePathHandleFunc returns a function for handling static HTML
func MakePathHandleFunc(pagePath string) HandleFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		HTML, err := ioutil.ReadFile(pagePath)
		if err != nil {
			Error.Println(err)
		}
		w.Write(HTML)
	}
}

func WrapHandler(handler http.Handler) HandleFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		Info.Printf("%s is requesting %s", r.RemoteAddr, r.URL)
		handler.ServeHTTP(w, r)
	}
}

func WrapHandleFunc(fn HandleFunc) HandleFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		Info.Printf("%s is requesting %s", r.RemoteAddr, r.URL)
		fn(w, r)
	}
}

func dashboardHandler(w http.ResponseWriter, r *http.Request) {
	// use template to insert assignment links
	tmpl, err := template.ParseFiles(
		path.Join(env.DashboardPath()))
	if err != nil {
		Error.Println(err)
		http.NotFound(w, r)
		return
	}
	projectName := r.FormValue("project_name")
	dashboardContents := DashboardContents{
		Project:     GetProject(projectName),
		Assignments: GetAssignmentsInProject(projectName),
	}
	Info.Println(dashboardContents.Assignments) // project is too verbose to log
	tmpl.Execute(w, dashboardContents)
}

func vendorHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.VendorPath())
	if err != nil {
		Error.Println(err)
		http.NotFound(w, r)
		return
	}
	projectName := r.FormValue("project_name")
	dashboardContents := DashboardContents{
		Project:     GetProject(projectName),
		Assignments: GetAssignmentsInProject(projectName),
	}
	Info.Println(dashboardContents.Assignments) // project is too verbose to log
	tmpl.Execute(w, dashboardContents)
}

// TODO: split this function up
// Handles the posting of new projects
func postProjectHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}
	var projectName = CheckProjectName(r.FormValue("project_name"))
	if projectName == "" {
		w.Write([]byte("Project Name already exists."))
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
	var vmd VideoMetaData
	if itemType == "video" {
		var frameDirectoryItems []Item
		err = yaml.Unmarshal(itemFileBuf.Bytes(), &frameDirectoryItems)
		if err != nil {
			Error.Println(err)
		}
		// in video, we only consider the first item, and
		//   the item url is the frame directory
		frameUrl := frameDirectoryItems[0].Url
		framePath := path.Join(env.DataDir, frameUrl[1:len(frameUrl)])
		// if no frames directory for this vid, throw error
		_, err := os.Stat(framePath)
		if err != nil {
			Error.Println(framePath + " does not exist. Has video been split into frames?")
			http.NotFound(w, r)
			return
		}
		// get the video's metadata
		mdContents, _ := ioutil.ReadFile(path.Join(framePath, "metadata.json"))
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
				Url:   path.Join(frameUrl, "f-"+frameString+".jpg"),
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
			items[i].Index = i
		}
	}

	// categories YAML
	var categories = getCategories(r)


	// attributes YAML
	var attributes = getAttributes(r)


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
		VideoMetaData: vmd,
	}
	// Save project to project folder
	projectPath := project.GetPath()
	projectJson, _ := json.MarshalIndent(project, "", "  ")
	err = ioutil.WriteFile(projectPath, projectJson, 0644)
	if err != nil {
		Error.Println("Failed to save project file of", project.Name)
	} else {
		Info.Println("Saving project file of", project.Name)
	}

	index := 0
	handlerUrl := GetHandlerUrl(project)
	pageTitle := r.FormValue("page_title")
	if err != nil {
		Error.Println(err)
	}
	if itemType == "video" {
		task := Task{
			HandlerUrl:    handlerUrl,
			PageTitle:     pageTitle,
			ProjectName:   project.Name,
			Index:         0,
			Items:         project.Items,
			Categories:    project.Categories,
			Attributes:    project.Attributes,
			VideoMetaData: vmd,
		}
		index = 1
		assignmentInfo := AssignmentInfo {
			SubmissionsCount: 0,
		}
		workerId, _ := strconv.Atoi(r.FormValue("worker_id"))
		assignment := Assignment {
			Task:     task,
			WorkerId: workerId,
			Info:     assignmentInfo,
		}

		// Save task to task folder
		taskPath := task.GetPath()
		taskJson, _ := json.MarshalIndent(task, "", "  ")
		err = ioutil.WriteFile(taskPath, taskJson, 0644)
		if err != nil {
			Error.Println("Failed to save task file of", task.ProjectName,
				task.Index)
		} else {
			Info.Println("Saving task file of", task.ProjectName,
				task.Index)
		}
		// Save assignment to assignment folder
		assignmentPath := assignment.GetPath()
		assignmentJson, _ := json.MarshalIndent(assignment, "", "  ")
		err = ioutil.WriteFile(assignmentPath, assignmentJson, 0644)
		if err != nil {
			Error.Println("Failed to save assignment file of",
				assignment.Task.ProjectName, assignment.Task.Index)
		} else {
			Info.Println("Saving assignment file of",
				assignment.Task.ProjectName, assignment.Task.Index)
		}
	} else {
		size := len(project.Items)
		for i := 0; i < size; i += taskSize {
			// Initialize new task
			task := Task{
				HandlerUrl:  handlerUrl,
				PageTitle:   pageTitle,
				ProjectName: project.Name,
				Index:       index,
				Items:       project.Items[i:Min(i+taskSize, size)],
				Categories:  project.Categories,
				Attributes:  project.Attributes,
			}
			index = index + 1
			assignmentInfo := AssignmentInfo{
				SubmissionsCount: 0,
			}
			workerId, _ := strconv.Atoi(r.FormValue("worker_id"))
			assignment := Assignment{
				Task:     task,
				WorkerId: workerId,
				Info:     assignmentInfo,
			}

			// Save task to task folder
			taskPath := task.GetPath()
			taskJson, _ := json.MarshalIndent(task, "", "  ")
			err = ioutil.WriteFile(taskPath, taskJson, 0644)
			if err != nil {
				Error.Println("Failed to save task file of", task.ProjectName,
					task.Index)
			} else {
				Info.Println("Saving task file of", task.ProjectName,
					task.Index)
			}
			// Save assignment to assignment folder
			assignmentPath := assignment.GetPath()
			assignmentJson, _ := json.MarshalIndent(assignment, "", "  ")
			err = ioutil.WriteFile(assignmentPath, assignmentJson, 0644)
			if err != nil {
				Error.Println("Failed to save assignment file of",
					assignment.Task.ProjectName, assignment.Task.Index)
			} else {
				Info.Println("Saving assignment file of",
					assignment.Task.ProjectName, assignment.Task.Index)
			}
		}
	}

	Info.Println("Created", index, "new tasks")

	// TODO: is this necessary?
	// w.Write([]byte(strconv.Itoa(index)))
}

// Handles the loading of an assignment given its project name and task index.
func postLoadAssignmentHandler(w http.ResponseWriter, r *http.Request) {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		Error.Println(err)
	}
	assignmentToLoad := Assignment{}
	err = json.Unmarshal(body, &assignmentToLoad)
	if err != nil {
		Error.Println(err)
	}
	loadedAssignment := GetAssignment(assignmentToLoad.Task.ProjectName,
		strconv.Itoa(assignmentToLoad.Task.Index))
	loadedAssignmentJson, err := json.Marshal(loadedAssignment)
	if err != nil {
		Error.Println(err)
	}
	w.Write(loadedAssignmentJson)
}

// Handles the posting of saved assignments
func postSaveHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		Error.Println(err)
	}
	assignment := Assignment{}
	err = json.Unmarshal(body, &assignment)
	if err != nil {
		Error.Println(err)
	}
	Info.Println(assignment)

	oldAssignment := GetAssignment(assignment.Task.ProjectName, strconv.Itoa(assignment.Task.Index))
	assignment.Events = append(oldAssignment.Events, assignment.Events...)

	assignmentPath := path.Join(env.DataDir, "assignments",
	    assignment.Task.ProjectName, strconv.Itoa(assignment.Task.Index) +
	    ".json")
	assignmentJson, err := json.MarshalIndent(assignment, "", "  ")
	if err != nil {
		Error.Println(err)
	}

	err = ioutil.WriteFile(assignmentPath, assignmentJson, 0644)
	if err != nil {
		Error.Println(err)
	} else {
		Info.Println("Saved assignment " + assignmentPath)
	}

	w.Write(nil)
}

// Handles the posting of completed tasks
func postSubmissionHandler(w http.ResponseWriter, r *http.Request) {
	// TODO
}

// handles category YAML file, sets to default values if file missing
func getCategories(r *http.Request) []Category {
	labelType := r.FormValue("label_type")
	var categories []Category
	categoryFile, _, err := r.FormFile("categories")

	switch err {
	case nil:
		defer categoryFile.Close()

		categoryFileBuf := bytes.NewBuffer(nil)
		_, err = io.Copy(categoryFileBuf, categoryFile)
		if err != nil {
			Error.Println(err)
		}
		err = yaml.Unmarshal(categoryFileBuf.Bytes(), &categories)
		if err != nil {
			Error.Println(err)
		}

	case http.ErrMissingFile:
		Info.Printf("Miss category file and using default categories for %s.", labelType)

		if labelType == "box2d" {
			categories = defaultBox2dCategories
		} else if labelType == "segmentation" {
			categories = defaultSeg2dCategories
		} else if labelType == "lane" {
			categories = defaultLane2dCategories
		} else {
			Error.Printf("No default categories for %s.", labelType)
		}

	default:
		log.Println(err)
	}

	return categories
}

// handles category YAML file, sets to default values if file missing
func getAttributes(r *http.Request) []Attribute {
	labelType := r.FormValue("label_type")
	var attributes []Attribute
	attributeFile, _, err := r.FormFile("attributes")

	switch err {
	case nil:
		defer attributeFile.Close()

		attributeFileBuf := bytes.NewBuffer(nil)
		_, err = io.Copy(attributeFileBuf, attributeFile)
		if err != nil {
			Error.Println(err)
		}
		err = yaml.Unmarshal(attributeFileBuf.Bytes(), &attributes)
		if err != nil {
			Error.Println(err)
		}

	case http.ErrMissingFile:
		Info.Printf("Missing attribute file and using default attributes for %s.", labelType)

		if labelType == "box2d" {
			attributes = defaultBox2dAttributes
		} else {
			attributes = dummyAttribute
			Info.Printf("No default attributes for %s.", labelType)
		}

	default:
		log.Println(err)
	}

	return attributes
}
