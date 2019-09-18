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
	"strconv"

	"github.com/mitchellh/mapstructure"
)

//Sat state
type Sat struct {
	Task    TaskData    `json:"task" yaml:"task"`
	User    UserData    `json:"user" yaml:"user"`
	Session SessionData `json:"session" yaml:"session"`
}

//Task specific data
type TaskData struct {
	Config ConfigData `json:"config" yaml:"config"`
	Status TaskStatus `json:"status" yaml:"status"`
	Items  []ItemData `json:"items" yaml:"items"`
	Tracks TrackMap   `json:"tracks" yaml:"tracks"`
}

//Task properties not changed during lifetime of a session
type ConfigData struct {
	ProjectName     string      `json:"projectName" yaml:"projectName"`
	ItemType        string      `json:"itemType" yaml:"itemType"`
	LabelTypes      []string    `json:"labelTypes" yaml:"labelTypes"`
	TaskSize        int         `json:"taskSize" yaml:"taskSize"`
	HandlerUrl      string      `json:"handlerUrl" yaml:"handlerUrl"`
	PageTitle       string      `json:"pageTitle" yaml:"pageTitle"`
	InstructionPage string      `json:"instructionPage" yaml:"instructionPage"`
	BundleFile      string      `json:"bundleFile" yaml:"bundleFile"`
	Categories      []string    `json:"categories" yaml:"categories"`
	Attributes      []Attribute `json:"attributes" yaml:"attributes"`
	TaskId          string      `json:"taskId" yaml:"taskId"`
	SubmitTime      int64       `json:"submitTime" yaml:"submitTime"`
}

//Task properties that depend on the current state of session
type TaskStatus struct {
	MaxLabelId int `json:"maxLabelId" yaml:"maxLabelId"`
	MaxShapeId int `json:"maxShapeId" yaml:"maxShapeId"`
	MaxOrder   int `json:"maxOrder" yaml:"maxOrder"`
}

//Contains data for single item
type ItemData struct {
	Id     int               `json:"id" yaml:"id"`
	Index  int               `json:"index" yaml:"index"`
	Url    string            `json:"url" yaml:"url"`
	Labels map[int]LabelData `json:"labels" yaml:"labels"`
	Shapes map[int]ShapeData `json:"shapes" yaml:"shapes"`
}

//Contains data for single label
type LabelData struct {
	Id         int              `json:"id" yaml:"id"`
	Item       int              `json:"item" yaml:"item"`
	Type       string           `json:"type" yaml:"type"`
	Category   []int            `json:"category" yaml:"category"`
	Attributes map[string][]int `json:"attributes" yaml:"attributes"`
	Parent     int              `json:"parent" yaml:"parent"`
	Children   []int            `json:"children" yaml:"children"`
	Shapes     []int            `json:"shapes" yaml:"shapes"`
	Track      int              `json:"track" yaml:"track"`
	Order      int              `json:"order" yaml:"order"`
	Manual     bool             `json:"manual" yaml:"manual"`
}

//Contains data for single shape
type ShapeData struct {
	Id    int         `json:"id" yaml:"id"`
	Label []int       `json:"label" yaml:"label"`
	Type  string      `json:"type" yaml:"type"`
	Shape interface{} `json:"shape" yaml:"shape"`
}

//Data for tracks
type TrackMap map[int]interface{}

//User specific data
type UserData struct {
	UserId               string                 `json:"id" yaml:"id"`
	Selection            SelectedData           `json:"select" yaml:"select"`
	Layout               LayoutData             `json:"layout" yaml:"layout"`
	ImageViewConfig      ImageViewerConfig      `json:"imageViewerConfig" yaml:"imageViewerConfig"`
	PointCloudViewConfig PointCloudViewerConfig `json:"pointCloudViewerConfig" yaml:"pointCloudViewerConfig"`
}

//User's currently selected data
type SelectedData struct {
	Item      int `json:"item" yaml:"item"`
	Label     int `json:"label" yaml:"label"`
	Shape     int `json:"shape" yaml:"shape"`
	Category  int `json:"category" yaml:"category"`
	LabelType int `json:"labelType" yaml:"labelType"`
}

//Data for frontend layout
type LayoutData struct {
	ToolbarWidth       int     `json:"toolbarWidth" yaml:"toolbarWidth"`
	AssistantView      bool    `json:"assistantView" yaml:"assistantView"`
	AssistantViewRatio float32 `json:"assistantViewRatio" yaml:"assistantViewRatio"`
}

type ImageViewerConfig struct {
	ImageWidth  int     `json:"imageWidth" yaml:"imageWidth"`
	ImageHeight int     `json:"imageHeight" yaml:"imageHeight"`
	ViewScale   float32 `json:"viewScale" yaml:"viewScale"`
	DisplayTop  int     `json:"displayTop" yaml:"displayTop"`
	DisplayLeft int     `json:"displayLeft" yaml:"displayLeft"`
}

type Vector3D struct {
	X float32 `json:"x" yaml:"x"`
	Y float32 `json:"y" yaml:"y"`
	Z float32 `json:"z" yaml:"z"`
}

type PointCloudViewerConfig struct {
	Target       Vector3D `json:"target" yaml:"target"`
	Position     Vector3D `json:"position" yaml:"position"`
	VerticalAxis Vector3D `json:"verticalAxis" yaml:"verticalAxis"`
}

//Session specific data
type SessionData struct {
	SessionId    string       `json:"id" yaml:"id"`
	DemoMode     bool         `json:"demoMode" yaml:"demoMode"`
	StartTime    int64        `json:"startTime" yaml:"startTime"`
	ItemStatuses []ItemStatus `json:"items" yaml:"items"`
}

//Item status
type ItemStatus struct {
	Loaded bool `json:"loaded" yaml:"loaded"`
}

//GetKey gets key for single task and worker
func (sat *Sat) GetKey() string {
	return path.Join(sat.Task.Config.ProjectName, "submissions",
		sat.Task.Config.TaskId, sat.User.UserId,
		strconv.FormatInt(sat.Task.Config.SubmitTime, 10))
}

//GetFields returns sat as a map
func (sat *Sat) GetFields() map[string]interface{} {
	return map[string]interface{}{
		"task":    sat.Task,
		"user":    sat.User,
		"session": sat.Session,
	}
}

//GetSat gets the most recent assignment given the needed fields.
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
		err = mapstructure.Decode(fields, &assignment)
		if err != nil {
			return Sat{}, err
		}
		sat = assignmentToSat(&assignment)
	}
	return sat, nil
}

//GetAssignmentV2 retrieves assignment
func GetAssignmentV2(projectName string, taskIndex string,
	workerId string) (Assignment, error) {
	assignment := Assignment{}
	assignmentPath := path.Join(projectName, "assignments", taskIndex, workerId)
	fields, err := storage.Load(assignmentPath)
	if err != nil {
		return Assignment{}, err
	}
	err = mapstructure.Decode(fields, &assignment)
	if err != nil {
		return Assignment{}, err
	}
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
		taskIndex, DefaultWorker)) {
		// if assignment does not exist, create it
		loadedAssignment, err = CreateAssignment(projectName, taskIndex,
			DefaultWorker)
		if err != nil {
			Error.Println(err)
			return
		}
		loadedSat = assignmentToSat(&loadedAssignment)
	} else {
		loadedSat, err = GetSat(projectName, taskIndex,
			DefaultWorker)
		if err != nil {
			Error.Println(err)
			return
		}
	}
	loadedSat.Session.StartTime = recordTimestamp()
	loadedSatJson, err := json.Marshal(loadedSat)
	if err != nil {
		Error.Println(err)
	}
	_, err = w.Write(loadedSatJson)
	if err != nil {
		Error.Println(err)
	}
}

func executeLabelingTemplateV2(w http.ResponseWriter, r *http.Request,
	tmpl *template.Template) {
	// get task name from the URL
	projectName := r.URL.Query()["project_name"][0]
	taskIndex, _ := strconv.ParseInt(r.URL.Query()["task_index"][0], 10, 32)
	if !storage.HasKey(path.Join(projectName, "assignments",
		Index2str(int(taskIndex)), DefaultWorker)) {
		// if assignment does not exist, create it
		assignment, err := CreateAssignment(projectName,
			Index2str(int(taskIndex)), DefaultWorker)
		if err != nil {
			Error.Println(err)
			return
		}
		err = tmpl.Execute(w, assignment)
		if err != nil {
			Error.Println(err)
		}
	} else {
		// otherwise, get that assignment
		assignment, err := GetAssignmentV2(projectName,
			Index2str(int(taskIndex)), DefaultWorker)
		if err != nil {
			Error.Println(err)
			return
		}
		err = tmpl.Execute(w, assignment)
		if err != nil {
			Error.Println(err)
		}
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
	var items []ItemData
	var itemStatuses []ItemStatus
	for _, item := range assignment.Task.Items {
		satItem := ItemData{
			Id:     item.Index,
			Index:  item.Index,
			Url:    item.Url,
			Labels: map[int]LabelData{},
			Shapes: map[int]ShapeData{},
		}
		itemStatus := ItemStatus{
			Loaded: false,
		}
		items = append(items, satItem)
		itemStatuses = append(itemStatuses, itemStatus)
	}
	// only items are needed because this function is only called once
	// at the first visit to annotation interface before submission
	// and will go away when redux have its own project creation logic
	projectOptions := assignment.Task.ProjectOptions

	configData := ConfigData{
		ProjectName:     projectOptions.Name,
		ItemType:        projectOptions.ItemType,
		LabelTypes:      []string{projectOptions.LabelType},
		TaskSize:        projectOptions.TaskSize,
		HandlerUrl:      projectOptions.HandlerUrl,
		PageTitle:       projectOptions.PageTitle,
		InstructionPage: projectOptions.Instructions,
		BundleFile:      projectOptions.BundleFile,
		Categories:      categories,
		Attributes:      projectOptions.Attributes,
		TaskId:          Index2str(assignment.Task.Index),
		SubmitTime:      assignment.SubmitTime,
	}

	taskData := TaskData{
		Config: configData,
		Items:  items,
		Tracks: TrackMap{},
	}

	selectedData := SelectedData{
		Item:  0,
		Label: 0,
	}

	imageViewConfig := ImageViewerConfig{
		ImageWidth:  0,
		ImageHeight: 0,
		ViewScale:   1.0,
		DisplayLeft: 0,
		DisplayTop:  0,
	}

	target := Vector3D{
		X: 0.,
		Y: 0.,
		Z: 0.,
	}

	position := Vector3D{
		X: 0.,
		Y: 10.,
		Z: 0.,
	}

	verticalAxis := Vector3D{
		X: 0.,
		Y: 0.,
		Z: 1.,
	}

	pointCloudViewConfig := PointCloudViewerConfig{
		Target:       target,
		Position:     position,
		VerticalAxis: verticalAxis,
	}

	userData := UserData{
		UserId:               assignment.WorkerId,
		Selection:            selectedData,
		ImageViewConfig:      imageViewConfig,
		PointCloudViewConfig: pointCloudViewConfig,
	}

	uuid := getUuidV4()
	sessionData := SessionData{
		SessionId:    uuid,
		DemoMode:     projectOptions.DemoMode,
		StartTime:    assignment.StartTime,
		ItemStatuses: itemStatuses,
	}

	loadedSat := Sat{
		Task:    taskData,
		User:    userData,
		Session: sessionData,
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
		writeNil(w)
		return
	}
	if assignment.Session.DemoMode {
		Error.Println(errors.New("can't save a demo project"))
		writeNil(w)
		return
	}

	assignment.Task.Config.SubmitTime = recordTimestamp()
	err = storage.Save(assignment.GetKey(), assignment.GetFields())
	if err != nil {
		Error.Println(err)
		writeNil(w)
	} else {
		response, err := json.Marshal(0)
		if err != nil {
			Error.Println(err)
			writeNil(w)
		} else {
			_, err = w.Write(response)
			if err != nil {
				Error.Println(err)
			}
		}
	}
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
	err = mapstructure.Decode(fields, &projectToLoad)
	if err != nil {
		Error.Println(err)
	}
	// Grab the latest submissions from all tasks
	tasks, err := GetTasksInProject(projectName)
	if err != nil {
		Error.Println(err)
		return
	}
	items := []ItemExportV2{}
	sat := Sat{}
	for _, task := range tasks {
		sat, err = GetSat(projectName, Index2str(task.Index), DefaultWorker)
		if err == nil {
			for _, itemToLoad := range sat.Task.Items {
				item := exportItemData(
					itemToLoad,
					sat.Task.Config,
					task.Index,
					projectToLoad.Options.ItemType,
					projectToLoad.Options.Name)
				items = append(items, item)
			}
		} else {
			// if file not found, return list of items with url
			Info.Println(err)
			for _, itemToLoad := range task.Items {
				item := ItemExportV2{}
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
	_, err = io.Copy(w, bytes.NewReader(exportJson))
	if err != nil {
		Error.Println(err)
	}

}
