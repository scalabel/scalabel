package main

import (
	"bytes"
	"encoding/json"
	"html/template"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"strconv"
	"strings"
	"gopkg.in/yaml.v2"
)

// Project is what the admin creates, specifying a list of items
type Project struct {
	Name          string        `json:"name"`
	ItemType      string        `json:"itemType"`
	LabelType     string        `json:"labelType"`
	Items         []Item        `json:"items"`
	Categories    []Category    `json:"categories"`
	TaskSize      int           `json:"taskSize"`
	Attributes    []Attribute   `json:"attributes"`
	VendorId      int           `json:"vendorId"`
	VideoMetadata VideoMetadata `json:"metadata"`
}

// A chunk of a project
type Task struct {
    ProjectName string
    Index       int
	Items       []Item
	Attributes  []Attribute
}

// The actual assignment of a task to an annotator
type Assignment struct {
	Task      Task              `json:""`
	WorkerId  int               `json:""`
	Events    []Event           `json:"events"`
	UserAgent string            `json:"userAgent"`
	IpInfo    map[string]string `json:ipInfo"`
}

// An item is something to be annotated e.g. Image, PointCloud
type Item struct {
	Url         string  `json:"url" yaml:"url"`
	Index       int     `json:"index"`
	LabelIds    []int   `json:"labels"`
	GroundTruth []Label `json:"groundTruth" yaml:"groundTruth"`
}

// An annotation for an item, needs to include all possible annotation types
type Label struct {
	Id               int                `json:"id"`
	Category         Category           `json:"name"`
	ParentId         int                `json:"parent"`
	ChildrenIds      []int              `json:"children"`
	AttributeValues  map[string]bool    `json:"attributeValues"`
	Box2d            map[string]float32 `json:"box2d"`
}

// A class value for a label.
type Category struct {
	Name          string     `yaml:"name"`
	Subcategories []Category `yaml:"subcategories"`
}

// A configurable attribute describing a label
type Attribute struct {
	Name         string   `yaml:"name"`
	ToolType     string   `yaml:"toolType"`
	TagText      string   `yaml:"tagText"`
	TagPrefix    string   `yaml:"tagPrefix"`
	TagSuffixes  string   `yaml:"tagSuffixes"`
	Values       []string `yaml:"values"`
	ButtonColors []string `yaml:"buttonColors"`
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
	tmpl.Execute(w, GetTasks())
}

// Handles the posting of new projects
func postProjectHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}

	// item list YAML
	itemType := r.FormValue("item_type")
	var items []Item
	if itemType == "video" {
		videoName := r.FormValue("video_name")
		// check video has been converted to frames
		videoPath := env.DataDir + "/videos/" + videoName
		framePath := env.DataDir + "/frames/" + videoName
		videoPath = videoPath[:len(videoPath)-4] // take off the .mp4
		framePath = framePath[:len(framePath)-4]
		// if no frames directory for this vid, throw error
		_, err := os.Stat(framePath)
		if err != nil {
			Error.Println(videoPath + " has not been split into frames.")
			http.NotFound(w, r)
			return
		}
		// get the video's metadata
		mdContents, _ := ioutil.ReadFile(framePath + "/metadata.json")
		vmd := VideoMetadata{}
		json.Unmarshal(mdContents, &vmd)
		// get the URLs of all frame images
		numFrames, err := strconv.Atoi(vmd.NumFrames)
		if err != nil {
			Error.Println(err)
		}
		for i:=0; i < numFrames; i++ {
			frameString := strconv.Itoa(i + 1)
			for len(frameString) < 7 {
				frameString = "0" + frameString
			}
			frameItem := Item{
				Url: "./frames/" + videoName[:len(videoName) - 4] + "/" + frameString + ".jpg",
				Index: i,
			}
			items = append(items, frameItem)
		}
	} else {
		itemFile, _, err := r.FormFile("item_list")
		defer itemFile.Close()
		if err != nil {
			Error.Println(err)
		}
		itemFileBuf := bytes.NewBuffer(nil)
		_, err = io.Copy(itemFileBuf, itemFile)
		if err != nil {
			Error.Println(err)
		}
		err = yaml.Unmarshal(itemFileBuf.Bytes(), items)
		if err != nil {
			Error.Println(err)
		}
	}

	// categories YAML
	var categories []Category
	categoryFile, _, err := r.FormFile("categories")
	if err != nil {
		Error.Println(err)
	}
	categoryFileBuf := bytes.NewBuffer(nil)
	_, err = io.Copy(categoryFileBuf, categoryFile)
	if err != nil {
		Error.Println(err)
	}
	err = yaml.Unmarshal(categoryFileBuf.Bytes(), categories)
	if err != nil {
		Error.Println(err)
	}

	// attributes YAML
	var attributes []Attribute
	attributeFile, _, err := r.FormFile("custom_attributes")
	if err != nil {
		Error.Println(err)
	}
	attributeFileBuf := bytes.NewBuffer(nil)
	_, err = io.Copy(attributeFileBuf, attributeFile)
	if err != nil {
		Error.Println(err)
	}
	err = yaml.Unmarshal(attributeFileBuf.Bytes(), attributes)
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
		Name:       r.FormValue("project_name"),
		ItemType:   r.FormValue("item_type"),
		LabelType:  r.FormValue("label_type"),
		Items:      items,
		Categories: categories,
		TaskSize:   taskSize,
		Attributes: attributes,
		VendorId:   vendorId,
	}

	index := 0
	if itemType == "video" {
		task := Task{
			ProjectName: project.Name,
			Index: 0,
			Items: project.Items,
			Attributes: project.Attributes,
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
		for i:=0; i < size; i += taskSize {
			// Initialize new task
			task := Task{
				ProjectName: project.Name,
				Index:       index,
				Items:       project.Items[i:Min(i+taskSize, size)],
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

// Handles the posting of saved tasks
func postSaveHandler(w http.ResponseWriter, r *http.Request) {
	// TODO
}

// Handles the posting of completed tasks
func postSubmissionHandler(w http.ResponseWriter, r *http.Request) {
	// TODO
}
