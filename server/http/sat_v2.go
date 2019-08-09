package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"io/ioutil"
	"net/http"
	"path"
	"reflect"
	"strconv"

	"github.com/mitchellh/mapstructure"
)

//TOOD: correctly process all interface types

//Sat state
type Sat struct {
	Config  SatConfig  `json:"config" yaml:"config"`
	Current SatCurrent `json:"current" yaml:"current"`
	Items   []SatItem  `json:"items" yaml:"items"`
	Tracks  TrackMap   `json:"tracks" yaml:"tracks"`
	Layout  SatLayout  `json:"layout" yaml:"layout"`
}

//Sat configuration
type SatConfig struct {
	SessionId       string      `json:"sessionId" yaml:"sessionId"`
	AssignmentId    string      `json:"assignmentId" yaml:"assignmentId"`
	ProjectName     string      `json:"projectName" yaml:"projectName"`
	ItemType        string      `json:"itemType" yaml:"itemType"`
	LabelTypes      []string    `json:"labelTypes" yaml:"labelTypes"`
	TaskSize        int         `json:"taskSize" yaml:"taskSize"`
	HandlerUrl      string      `json:"handlerUrl" yaml:"handlerUrl"`
	PageTitle       string      `json:"pageTitle" yaml:"pageTitle"`
	InstructionPage string      `json:"instructionPage" yaml:"instructionPage"`
	DemoMode        bool        `json:"demoMode" yaml:"demoMode"`
	BundleFile      string      `json:"bundleFile" yaml:"bundleFile"`
	Categories      []string    `json:"categories" yaml:"categories"`
	Attributes      []Attribute `json:"attributes" yaml:"attributes"`
	TaskId          string      `json:"taskId" yaml:"taskId"`
	WorkerId        string      `json:"workerId" yaml:"workerId"`
	StartTime       int64       `json:"startTime" yaml:"startTime"`
	SubmitTime      int64       `json:"submitTime" yaml:"submitTime"`
}

//current state of Sat
type SatCurrent struct {
	Item       int `json:"item" yaml:"item"`
	Label      int `json:"label" yaml:"label"`
	Shape      int `json:"shape" yaml:"shape"`
	Category   int `json:"category" yaml:"category"`
	LabelType  int `json:"labelType" yaml:"labelType"`
	MaxLabelId int `json:"maxLabelId" yaml:"maxLabelId"`
	MaxShapeId int `json:"maxShapeId" yaml:"maxShapeId"`
	MaxOrder   int `json:"maxOrder" yaml:"maxOrder"`
}

type SatItem struct {
	Id           int              `json:"id" yaml:"id"`
	Index        int              `json:"index" yaml:"index"`
	Url          string           `json:"url" yaml:"url"`
	Active       bool             `json:"active" yaml:"active"`
	Loaded       bool             `json:"loaded" yaml:"loaded"`
	Labels       map[int]SatLabel `json:"labels" yaml:"labels"`
	Shapes       map[int]SatShape `json:"shapes" yaml:"shapes"`
	ViewerConfig interface{}      `json:"viewerConfig" yaml:"viewerConfig"`
}

type SatLabel struct {
	Id            int              `json:"id" yaml:"id"`
	Item          int              `json:"item" yaml:"item"`
	Type          string           `json:"type" yaml:"type"`
	Category      []int            `json:"category" yaml:"category"`
	Attributes    map[string][]int `json:"attributes" yaml:"attributes"`
	Parent        int              `json:"parent" yaml:"parent"`
	Children      []int            `json:"children" yaml:"children"`
	Shapes        []int            `json:"shapes" yaml:"shapes"`
	SelectedShape int              `json:"selectedShape" yaml:"selectedShape"`
	State         int              `json:"state" yaml:"state"`
	Order         int              `json:"order" yaml:"order"`
}

type SatShape struct {
	Id    int         `json:"id" yaml:"id"`
	Label []int       `json:"label" yaml:"label"`
	Shape interface{} `json:"shape" yaml:"shape"`
}

type TrackMap map[int]interface{}

type SatLayout struct {
	ToolbarWidth       int     `json:"toolbarWidth" yaml:"toolbarWidth"`
	AssistantView      bool    `json:"assistantView" yaml:"assistantView"`
	AssistantViewRatio float32 `json:"assistantViewRatio" yaml:"assistantViewRatio"`
}

func (sat *Sat) GetKey() string {
	return path.Join(sat.Config.ProjectName, "submissions", sat.Config.TaskId,
		sat.Config.WorkerId, strconv.FormatInt(sat.Config.SubmitTime, 10))
}

func (sat *Sat) GetFields() map[string]interface{} {
	return map[string]interface{}{
		"config":  sat.Config,
		"current": sat.Current,
		"items":   sat.Items,
		"tracks":  sat.Tracks,
	}
}

// Get the most recent assignment given the needed fields.
func GetSat(projectName string, taskIndex string,
	workerId string) (Sat, error) {
	sat := Sat{}
	submissionsPath := path.Join(projectName, "submissions",
		taskIndex, workerId)
	keys := storage.ListKeys(submissionsPath)
	// if any submissions exist, get the most recent one
	if len(keys) > 0 {
		Info.Printf("Reading %s\n", keys[len(keys)-1])
		fields, err := storage.Load(keys[len(keys)-1])
		if err != nil {
			return Sat{}, err
		}
		loadedSatJson, err := json.Marshal(fields)
		if err != nil {
			return Sat{}, err
		}
		if err := json.Unmarshal(loadedSatJson, &sat); err != nil {
			return Sat{}, err
		}
	} else {
		var assignment Assignment
		assignmentPath := path.Join(projectName, "assignments",
			taskIndex, workerId)
		Info.Printf("Reading %s\n", assignmentPath)
		fields, err := storage.Load(assignmentPath)
		if err != nil {
			return Sat{}, err
		}
		mapstructure.Decode(fields, &assignment)
		sat = assignmentToSat(&assignment)
	}
	return sat, nil
}

func GetAssignmentV2(projectName string, taskIndex string,
	workerId string) (Assignment, error) {
	assignment := Assignment{}
	assignmentPath := path.Join(projectName, "assignments", taskIndex, workerId)
	fields, err := storage.Load(assignmentPath)
	if err != nil {
		return Assignment{}, err
	}
	mapstructure.Decode(fields, &assignment)
	return assignment, nil
}

/* Handles the loading of an assignment given
   its project name, task index, and worker ID. */
func postLoadAssignmentV2Handler(w http.ResponseWriter, r *http.Request) {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		Error.Println(err)
	}
	assignmentToLoad := Assignment{}
	err = json.Unmarshal(body, &assignmentToLoad)
	if err != nil {
		Error.Println(err)
	}
	projectName := assignmentToLoad.Task.ProjectOptions.Name
	taskIndex := Index2str(assignmentToLoad.Task.Index)
	var loadedAssignment Assignment
	var loadedSat Sat
	if !storage.HasKey(path.Join(projectName, "assignments",
		taskIndex, DEFAULT_WORKER)) {
		// if assignment does not exist, create it
		loadedAssignment, err = CreateAssignment(projectName, taskIndex,
			DEFAULT_WORKER)
		if err != nil {
			Error.Println(err)
			return
		}
		loadedSat = assignmentToSat(&loadedAssignment)
	} else {
		loadedSat, err = GetSat(projectName, taskIndex,
			DEFAULT_WORKER)
		if err != nil {
			Error.Println(err)
			return
		}
	}
	loadedSat.Config.StartTime = recordTimestamp()
	loadedSatJson, err := json.Marshal(loadedSat)
	if err != nil {
		Error.Println(err)
	}
	w.Write(loadedSatJson)
}

func executeLabelingTemplateV2(w http.ResponseWriter, r *http.Request,
	tmpl *template.Template) {
	// get task name from the URL
	projectName := r.URL.Query()["project_name"][0]
	taskIndex, _ := strconv.ParseInt(r.URL.Query()["task_index"][0], 10, 32)
	if !storage.HasKey(path.Join(projectName, "assignments",
		Index2str(int(taskIndex)), DEFAULT_WORKER)) {
		// if assignment does not exist, create it
		assignment, err := CreateAssignment(projectName,
			Index2str(int(taskIndex)), DEFAULT_WORKER)
		if err != nil {
			Error.Println(err)
			return
		}
		tmpl.Execute(w, assignment)
	} else {
		// otherwise, get that assignment
		assignment, err := GetAssignmentV2(projectName,
			Index2str(int(taskIndex)), DEFAULT_WORKER)
		if err != nil {
			Error.Println(err)
			return
		}
		tmpl.Execute(w, assignment)
	}
}

func Label2dv2Handler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.Label2dPath(r.FormValue("v")))
	if err != nil {
		Error.Println(err)
	}
	executeLabelingTemplateV2(w, r, tmpl)
}

func Label3dv2Handler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.Label3dPath(r.FormValue("v")))
	if err != nil {
		Error.Println(err)
	}
	executeLabelingTemplate(w, r, tmpl)
}

// Essentially rewriting the decodeBaseJson logic, need to get rid of this
// when backend is completely transferred to redux
func assignmentToSat(assignment *Assignment) Sat {
	var categories []string
	for _, category := range assignment.Task.ProjectOptions.Categories {
		categories = append(categories, category.Name)
	}
	var items []SatItem
	for _, item := range assignment.Task.Items {
		satItem := SatItem{
			Id:     item.Index,
			Index:  item.Index,
			Url:    item.Url,
			Labels: map[int]SatLabel{},
			Shapes: map[int]SatShape{},
		}
		items = append(items, satItem)
	}
	// only items are needed because this function is only called once
	// at the first visit to annotation interface before submission
	// and will go away when redux have its own project creation logic
	tracks := TrackMap{}

	projectOptions := assignment.Task.ProjectOptions
	loadedSatConfig := SatConfig{
		AssignmentId:    assignment.Id,
		ProjectName:     projectOptions.Name,
		ItemType:        projectOptions.ItemType,
		LabelTypes:      []string{projectOptions.LabelType},
		TaskSize:        projectOptions.TaskSize,
		HandlerUrl:      projectOptions.HandlerUrl,
		PageTitle:       projectOptions.PageTitle,
		InstructionPage: projectOptions.Instructions,
		DemoMode:        projectOptions.DemoMode,
		BundleFile:      projectOptions.BundleFile,
		Categories:      categories,
		Attributes:      projectOptions.Attributes,
		TaskId:          Index2str(assignment.Task.Index),
		WorkerId:        assignment.WorkerId,
		StartTime:       assignment.StartTime,
		SubmitTime:      assignment.SubmitTime,
	}
	satCurrent := SatCurrent{
		Item:  -1,
		Label: -1,
	}
	loadedSat := Sat{
		Config:  loadedSatConfig,
		Current: satCurrent,
		Items:   items,
		Tracks:  tracks,
	}
	return loadedSat
}

func postSaveV2Handler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		Error.Println(err)
	}

	assignment := Sat{}
	err = json.Unmarshal(body, &assignment)
	if err != nil {
		Error.Println(err)
		w.Write(nil)
		return
	}
	if assignment.Config.DemoMode {
		Error.Println(errors.New("can't save a demo project"))
		w.Write(nil)
		return
	}

	assignment.Config.SubmitTime = recordTimestamp()
	err = storage.Save(assignment.GetKey(), assignment.GetFields())
	if err != nil {
		Error.Println(err)
		w.Write(nil)
	} else {
		response, err := json.Marshal(0)
		if err != nil {
			w.Write(nil)
		} else {
			w.Write(response)
		}
	}
}

// helper function for exports
func exportSatItem(
	itemToLoad SatItem,
	satAttributes []Attribute,
	taskIndex int,
	itemType string,
	projectName string) ItemExport {
	item := ItemExport{}
	item.Index = itemToLoad.Index
	if itemType == "video" {
		item.VideoName = projectName + "_" + Index2str(taskIndex)
	}
	item.Timestamp = 10000 // to be fixed
	item.Name = itemToLoad.Url
	item.Url = itemToLoad.Url
	item.Attributes = map[string]string{}
	if len(itemToLoad.Labels) > 0 {
		itemLabel := itemToLoad.Labels[0]
		keys := reflect.ValueOf(itemLabel.Attributes).MapKeys()
		strkeys := make([]string, len(keys))
		for i := 0; i < len(keys); i++ {
			strkeys[i] = keys[i].String()
		}
		for _, key := range strkeys {
			for _, attribute := range satAttributes {
				if attribute.Name == key {
					item.Attributes[key] =
						attribute.Values[itemLabel.Attributes[key][0]]
					break
				}
			}
		}
	}
	return item
}

// Handles the export of submitted assignments
func postExportV2Handler(w http.ResponseWriter, r *http.Request) {
	var projectName = r.FormValue("project_name")
	key := path.Join(projectName, "project")
	fields, err := storage.Load(key)
	if err != nil {
		Error.Println(err)
	}
	projectToLoad := Project{}
	mapstructure.Decode(fields, &projectToLoad)

	// Grab the latest submissions from all tasks
	tasks, err := GetTasksInProject(projectName)
	if err != nil {
		Error.Println(err)
		return
	}
	items := []ItemExport{}
	for _, task := range tasks {
		sat, err := GetSat(projectName, Index2str(task.Index), DEFAULT_WORKER)
		if err == nil {
			for _, itemToLoad := range sat.Items {
				item := exportSatItem(
					itemToLoad,
					sat.Config.Attributes,
					task.Index,
					projectToLoad.Options.ItemType,
					projectToLoad.Options.Name)
				items = append(items, item)
			}
		} else {
			// if file not found, return list of items with url
			Info.Println(err)
			for _, itemToLoad := range task.Items {
				item := ItemExport{}
				item.Index = itemToLoad.Index
				if projectToLoad.Options.ItemType == "video" {
					item.VideoName = projectToLoad.Options.Name +
						"_" + Index2str(task.Index)
				}
				item.Timestamp = 10000 // to be fixed
				item.Name = itemToLoad.Url
				item.Url = itemToLoad.Url
				items = append(items, item)
			}
		}
	}

	exportJson, err := json.MarshalIndent(items, "", "  ")
	if err != nil {
		Error.Println(err)
	}

	//set relevant header.
	w.Header().Set("Content-Disposition",
		fmt.Sprintf("attachment; filename=%s_results.json", projectName))
	io.Copy(w, bytes.NewReader(exportJson))
}
