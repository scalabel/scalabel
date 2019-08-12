package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"path"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/mitchellh/mapstructure"
	"gopkg.in/yaml.v2"
)

// Stores the user info
type User struct {
	Id           string   `json:"id"`
	Email        string   `json:"email"`
	Group        string   `json:"group"`
	RefreshToken string   `json:"refreshToken"`
	Projects     []string `json:"projects"`
}

//implements Serializable
type Project struct {
	Items    map[string][]Item `json:"items" yaml:"items"`
	VendorId int               `json:"vendorId" yaml:"vendorId"`
	Options  ProjectOptions    `json:"options" yaml:"options"`
}

func (project *Project) GetKey() string {
	return path.Join(project.Options.Name, "project")
}

func (project *Project) GetFields() map[string]interface{} {
	return map[string]interface{}{
		"Items":    project.Items,
		"VendorId": project.VendorId,
		"Options":  project.Options,
	}
}

//implements Serializable
type Task struct {
	ProjectOptions       ProjectOptions `json:"projectOptions" yaml:"projectOptions"`
	Index                int            `json:"index" yaml:"index"`
	Items                []Item         `json:"items" yaml:"items"`
	NumFrames            int            `json:"numFrames" yaml:"numFrames"`
	NumLabelImport       int            `json:"numLabelImport" yaml:"numLabelImport"`
	NumLabeledItemImport int            `json:"numLabeledItemImport" yaml:"numLabeledItemImport"`
}

func (task *Task) GetKey() string {
	return path.Join(task.ProjectOptions.Name, "tasks", Index2str(task.Index))
}

func (task *Task) GetFields() map[string]interface{} {
	return map[string]interface{}{
		"ProjectOptions":       task.ProjectOptions,
		"Index":                task.Index,
		"Items":                task.Items,
		"NumFrames":            task.NumFrames,
		"NumLabelImport":       task.NumLabelImport,
		"NumLabeledItemImport": task.NumLabeledItemImport,
	}
}

//implements Serializable
type Assignment struct {
	Id              string                 `json:"id" yaml:"id"`
	Task            Task                   `json:"task" yaml:"task"`
	WorkerId        string                 `json:"workerId" yaml:"workerId"`
	Labels          []Label                `json:"labels" yaml:"labels"`
	Tracks          []Label                `json:"tracks" yaml:"tracks"`
	Events          []Event                `json:"events" yaml:"events"`
	StartTime       int64                  `json:"startTime" yaml:"startTime"`
	SubmitTime      int64                  `json:"submitTime" yaml:"submitTime"`
	NumLabeledItems int                    `json:"numLabeledItems" yaml:"numLabeledItems"`
	UserAgent       string                 `json:"userAgent" yaml:"userAgent"`
	IpInfo          map[string]interface{} `json:"ipInfo" yaml:"ipInfo"`
}

type GatewayInfo struct {
	Addr string `json:"Addr"`
	Port string `json:"Port"`
}

// Page data sent from the frontend, used in sending dashboard contents
type PageData struct {
	Name string `json:"name"`
}

func (assignment *Assignment) GetKey() string {
	task := assignment.Task
	if assignment.SubmitTime == 0 {
		return path.Join(task.ProjectOptions.Name,
			"assignments", Index2str(task.Index), assignment.WorkerId)
	}
	return path.Join(task.ProjectOptions.Name,
		"submissions", Index2str(task.Index),
		assignment.WorkerId, strconv.FormatInt(assignment.SubmitTime, 10))
}

func (assignment *Assignment) GetFields() map[string]interface{} {
	return map[string]interface{}{
		"Id":              assignment.Id,
		"Task":            assignment.Task,
		"WorkerId":        assignment.WorkerId,
		"Labels":          assignment.Labels,
		"Tracks":          assignment.Tracks,
		"Events":          assignment.Events,
		"StartTime":       assignment.StartTime,
		"SubmitTime":      assignment.SubmitTime,
		"NumLabeledItems": assignment.NumLabeledItems,
		"UserAgent":       assignment.UserAgent,
		"IpInfo":          assignment.IpInfo,
	}
}

// Info about a Project shared by Project and Task.
type ProjectOptions struct {
	Name              string        `json:"name" yaml:"name"`
	ItemType          string        `json:"itemType" yaml:"itemType"`
	LabelType         string        `json:"labelType" yaml:"labelType"`
	HandlerUrl        string        `json:"handlerUrl" yaml:"handlerUrl"`
	PageTitle         string        `json:"pageTitle" yaml:"pageTitle"`
	Instructions      string        `json:"instructions" yaml:"instructions"`
	InterpolationMode string        `json:"interpolationMode" yaml:"interpolationMode"`
	BundleFile        string        `json:"bundleFile" yaml:"bundleFile"`
	TaskSize          int           `json:"taskSize" yaml:"taskSize"`
	NumLeafCategories int           `json:"numLeafCategories" yaml:"numLeafCategories"`
	DemoMode          bool          `json:"demoMode" yaml:"demoMode"`
	Submitted         bool          `json:"submitted" yaml:"submitted"`
	VideoMetaData     VideoMetaData `json:"videoMetaData" yaml:"videoMetaData"`
	Detections        []Detection   `json:"detections" yaml:"detections"`
	Categories        []Category    `json:"categories" yaml:"categories"`
	Attributes        []Attribute   `json:"attributes" yaml:"attributes"`
}

/* Contains meta data about project, used for dashboard contents. This
is better than using project options as this contains the minimal amount of
data needed for the dashboard. */
type ProjectMetaData struct {
	Name              string `json:"name"`
	ItemType          string `json:"itemType"`
	LabelType         string `json:"labelType"`
	TaskSize          int    `json:"taskSize"`
	NumItems          int    `json:"numItems"`
	NumLeafCategories int    `json:"numLeafCategories"`
	NumAttributes     int    `json:"numAttributes"`
}

/* Contains meta data about tasks, used for dashboard conents. This
is better than sending all of the tasks, as this contains the minimal amount
of data needed for the dashboard. */
type TaskMetaData struct {
	NumLabeledImages int    `json:"numLabeledImages"`
	NumLabels        int    `json:"numLabels"`
	Submitted        bool   `json:"submitted"`
	HandlerUrl       string `json:"handlerUrl"`
}

// An item is something to be annotated e.g. Image, PointCloud
type Item struct {
	Url         string                 `json:"url" yaml:"url"`
	Index       int                    `json:"index" yaml:"index"`
	LabelIds    []int                  `json:"labelIds" yaml:"labelIds"`
	GroundTruth []Label                `json:"groundTruth" yaml:"groundTruth"`
	Data        map[string]interface{} `json:"data" yaml:"data"`
	LabelImport []LabelExport          `json:"labelImport" yaml:"labelImport"`
	Attributes  map[string][]int       `json:"attributes" yaml:"attributes"`
	VideoName   string                 `json:"videoName" yaml:"videoName"`
	Timestamp   int64                  `json:"timestamp" yaml:"timestamp"`
}

// An annotation for an item, needs to include all possible annotation types
type Label struct {
	Id           int                    `json:"id" yaml:"id"`
	CategoryPath string                 `json:"categoryPath" yaml:"categoryPath"`
	ParentId     int                    `json:"parentId" yaml:"parentId"`
	ChildrenIds  []int                  `json:"childrenIds" yaml:"childrenIds"`
	Attributes   map[string]interface{} `json:"attributes" yaml:"attributes"`
	Data         map[string]interface{} `json:"data" yaml:"data"`
	Keyframe     bool                   `json:"keyframe" yaml:"keyframe"`
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
	ProjectMetaData ProjectMetaData `json:"projectMetaData" yaml:"projectMetaData"`
	TaskMetaDatas   []TaskMetaData  `json:"taskMetaDatas" yaml:"taskMetaDatas"`
}

type TaskUrl struct { //shared type
	Url string `json:"url" yaml:"url"`
}

// unescaped marshal used to encode url string
func JsonMarshal(t interface{}) ([]byte, error) {
	buffer := &bytes.Buffer{}
	encoder := json.NewEncoder(buffer)
	encoder.SetEscapeHTML(false)
	err := encoder.Encode(t)
	return buffer.Bytes(), err
}

// Function type for handlers
type HandleFunc func(http.ResponseWriter, *http.Request)

func WrapHandler(handler http.Handler) HandleFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		Info.Printf("%s is requesting %s", r.RemoteAddr, r.URL)
		handler.ServeHTTP(w, r)
	}
}

func WrapHandleFunc(fn HandleFunc) HandleFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// check if User Management System is On
		flag := env.UserManagement == "on" ||
			env.UserManagement == "On" || env.UserManagement == "ON"
		refreshTokenCookie, _ := r.Cookie("refreshTokenScalabel")
		idCookie, _ := r.Cookie("idScalabel")
		if !flag { // if User Management System is off, continue
			fn(w, r)
			return
		} else if refreshTokenCookie == nil {
			redirectToLogin(w, r, "No refreshTokenCookie")
			return
		} else if idCookie == nil {
			redirectToLogin(w, r, "No idCookie")
			return
		} else if !verifyRefreshToken(refreshTokenCookie.Value,
			idCookie.Value) {
			redirectToLogin(w, r, "Failed to verify: Invalid Tokens")
			return
		} else {
			Info.Printf("%s is requesting %s", r.RemoteAddr, r.URL)
			fn(w, r)
			return
		}
	}
}

func countCategories(categories []Category) int {
	count := 0
	for _, category := range categories {
		if len(category.Subcategories) > 0 {
			count += countCategories(category.Subcategories)
		} else {
			count++
		}
	}
	return count
}

func createHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.CreatePath())
	if err != nil {
		Error.Println(err)
		http.NotFound(w, r)
		return
	}

	existingProjects := GetExistingProjects()
	err = tmpl.Execute(w, existingProjects)
	if err != nil {
		Error.Println(err)
	}
}

func dashboardHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(path.Join(env.DashboardPath()))
	if err != nil {
		Error.Println(err)
		http.NotFound(w, r)
		return
	}
	err = tmpl.Execute(w, "")
	if err != nil {
		Error.Println(err)
	}
}

func vendorHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.VendorPath())
	if err != nil {
		Error.Println(err)
		http.NotFound(w, r)
		return
	}
	err = tmpl.Execute(w, "")
	if err != nil {
		Error.Println(err)
	}
}

// Handles the posting of new projects
func postProjectHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}

	// validate form fields that are required
	err := formValidation(w, r)
	if err != nil {
		Error.Println(err)
	}

	// get version from the URL
	version := "v1"
	if r.URL.Query()["v"] != nil {
		version = r.URL.Query()["v"][0]
	}
	Info.Println("Selecting Version ", version)

	// make sure the project name in the form is new
	var projectName = CheckProjectName(r.FormValue("project_name"))
	if projectName == "" {
		_, err = w.Write([]byte("Project Name already exists."))
		if err != nil {
			Error.Println(err)
		}
		return
	}
	// get item type from form
	itemType := r.FormValue("item_type")
	// get frame rate and interpolation mode from form only if this is a video
	var videoMetaData VideoMetaData
	interpolationMode := "linear"
	var detections []Detection
	if itemType == "video" {
		interpolationMode = r.FormValue("interpolation_mode")
	}
	// get label type from form
	labelType := r.FormValue("label_type")
	// postpend version to supported label type
	if labelType == "box2d" && version == "v2" {
		labelType = "box2dv2"
	} else if labelType == "box3d" && version == "v2" {
		labelType = "box3dv2"
	}
	// get page title from form
	pageTitle := r.FormValue("page_title")
	// parse the category list YML from form
	categories := getCategoriesFromProjectForm(r)
	numLeafCategories := countCategories(categories)
	// parse the attribute list YML from form
	attributes := getAttributesFromProjectForm(r)
	// import items and corresponding labels
	itemLists := getItemsFromProjectForm(r, attributes)

	//this field should no longer be used, NumFrames is now stored in Task
	/*if itemType == "video" {
		videoMetaData.NumFrames = strconv.Itoa(len(items))
	}*/
	// get the task size from form
	var taskSize int
	var ts int
	if itemType != "video" {
		ts, err = strconv.Atoi(r.FormValue("task_size"))
		if err != nil {
			Error.Println(err)
			return
		}
		taskSize = ts
	} else {
		taskSize = 1
	}

	if itemType == "pointcloud" || itemType == "pointcloudtracking" {
		items := itemLists[" "] // assume there is only one image list
		var coeffs [4]float64
		for i := 0; i < len(items); i++ {
			coeffs, err = parsePLYForGround(items[i].Url)
			if err == nil {
				if items[i].Data == nil {
					items[i].Data = make(map[string]interface{})
				}
				items[i].Data["groundCoefficients"] = coeffs
			} else {
				Error.Println(err)
			}
		}
	}

	// get the vendor ID from form
	vendorId, err := strconv.Atoi(r.FormValue("vendor_id"))
	if err != nil {
		if r.FormValue("vendor_id") == "" {
			vendorId = -1
		} else {
			Error.Println(err)
			return
		}
	}

	// retrieve the link to instructions from form
	instructions := r.FormValue("instructions")

	demoMode := r.FormValue("demo_mode") == "true"

	// This prefix determines which handler will deal with labeling sessions
	//   for this project. Uniquely determined by item type and label type.
	handlerUrl := GetHandlerUrl(itemType, labelType)

	// get which bundle to use depending on redux progress
	bundleFile := "image.js"
	if labelType == "tag" || labelType == "box2dv2" {
		bundleFile = "image_v2.js"
	}

	// initialize and save the project
	var projectOptions = ProjectOptions{
		Name:              projectName,
		ItemType:          itemType,
		LabelType:         labelType,
		TaskSize:          taskSize,
		HandlerUrl:        handlerUrl,
		PageTitle:         pageTitle,
		Categories:        categories,
		NumLeafCategories: numLeafCategories,
		Attributes:        attributes,
		Instructions:      instructions,
		DemoMode:          demoMode,
		VideoMetaData:     videoMetaData,
		InterpolationMode: interpolationMode,
		Detections:        detections,
		BundleFile:        bundleFile,
		Submitted:         false,
	}
	var project = Project{
		Items:    itemLists,
		VendorId: vendorId,
		Options:  projectOptions,
	}

	// Save project to project folder
	err = storage.Save(project.GetKey(), project.GetFields())
	if err != nil {
		Error.Println(err)
	}
	// Initialize all the tasks
	CreateTasks(project)
}

func executeLabelingTemplate(w http.ResponseWriter,
	r *http.Request, tmpl *template.Template) {
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
		assignment, err := GetAssignment(projectName,
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

// Handles the loading of an assignment given
// its project name, task index, and worker ID.
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
	projectName := assignmentToLoad.Task.ProjectOptions.Name
	taskIndex := Index2str(assignmentToLoad.Task.Index)
	var loadedAssignment Assignment
	if !storage.HasKey(path.Join(projectName, "assignments",
		taskIndex, DefaultWorker)) {
		// if assignment does not exist, create it
		loadedAssignment, err = CreateAssignment(projectName, taskIndex,
			DefaultWorker)
		if err != nil {
			Error.Println(err)
			return
		}
	} else {
		loadedAssignment, err = GetAssignment(projectName, taskIndex,
			DefaultWorker)
		if err != nil {
			Error.Println(err)
			return
		}
		loadedAssignment.StartTime = recordTimestamp()
	}
	loadedAssignmentJson, err := json.Marshal(loadedAssignment)
	if err != nil {
		Error.Println(err)
	}
	_, err = w.Write(loadedAssignmentJson)
	if err != nil {
		Error.Println(err)
	}
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
	var fields map[string]interface{}
	err = json.Unmarshal(body, &fields)
	if err != nil {
		Error.Println(err)
	}
	fields["SubmitTime"] = recordTimestamp()
	assignment := Assignment{}
	err = mapstructure.Decode(fields, &assignment)
	if err != nil {
		Error.Println(err)
	}
	if assignment.Task.ProjectOptions.DemoMode {
		Error.Println(errors.New("can't save a demo project"))
		writeNil(w)
		return
	}
	// TODO: don't send all events to front end,
	// and append these events to most recent
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
func postExportHandler(w http.ResponseWriter, r *http.Request) {
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
	items := []ItemExport{}
	var latestSubmission Assignment
	for _, task := range tasks {
		latestSubmission, err = GetAssignment(projectName,
			Index2str(task.Index), DefaultWorker)
		if err == nil {
			for _, itemToLoad := range latestSubmission.Task.Items {
				item := ItemExport{}
				item.Index = itemToLoad.Index
				if projectToLoad.Options.ItemType == "video" {
					item.VideoName = itemToLoad.VideoName
				} else {
					//TODO: ask about what to do here
					item.VideoName = itemToLoad.VideoName
					//item.VideoName = projectToLoad.Options.Name
					//+ "_" + Index2str(task.Index)
				}
				item.Timestamp = itemToLoad.Timestamp
				item.Name = itemToLoad.Url
				item.Url = itemToLoad.Url
				for _, labelId := range itemToLoad.LabelIds {
					var labelToLoad Label
					for _, label := range latestSubmission.Labels {
						if label.Id == labelId {
							labelToLoad = label
							break
						}
					}
					label := LabelExport{}
					label.Category = labelToLoad.CategoryPath
					label.Attributes = labelToLoad.Attributes
					switch projectToLoad.Options.LabelType {
					case "box2d":
						label.Box2d = ParseBox2d(labelToLoad.Data)
					case "box3d":
						label.Box3d = ParseBox3d(labelToLoad.Data)
					case "segmentation":
						label.Poly2d = ParsePoly2d(labelToLoad.Data)
					case "lane":
						label.Poly2d = ParsePoly2d(labelToLoad.Data)
					}
					label.ManualShape = true
					if projectToLoad.Options.ItemType == "video" {
						label.ManualShape = labelToLoad.Keyframe
						label.Id = labelToLoad.ParentId
					} else {
						label.ManualShape = true
						label.Id = labelId
					}
					item.Labels = append(item.Labels, label)
				}
				items = append(items, item)
			}
		} else {
			// if file not found, return list of items with url
			Info.Println(err)
			for _, itemToLoad := range task.Items {
				item := ItemExport{}
				item.Index = itemToLoad.Index
				if projectToLoad.Options.ItemType == "video" {
					item.VideoName = itemToLoad.VideoName
				} else {
					//TODO: ask about what to do here
					item.VideoName = itemToLoad.VideoName
				}
				item.Timestamp = itemToLoad.Timestamp
				item.Name = itemToLoad.Url
				item.Url = itemToLoad.Url
				item.Labels = itemToLoad.LabelImport
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

// Handles the download of submitted assignments
func downloadTaskUrlHandler(w http.ResponseWriter, r *http.Request) {
	var projectName = r.FormValue("project_name")
	tasks, err := GetTasksInProject(projectName)
	if err != nil {
		Error.Println(err)
		return
	}

	var u *url.URL
	taskUrls := []TaskUrl{}
	for _, task := range tasks {
		taskUrl := TaskUrl{}
		u, err = url.Parse(task.ProjectOptions.HandlerUrl)
		if err != nil {
			log.Fatal(err)
		}
		q := u.Query()
		q.Set("project_name", projectName)
		q.Set("task_index", Index2str(task.Index))
		u.RawQuery = q.Encode()
		if r.TLS != nil {
			u.Scheme = "https"
		} else {
			u.Scheme = "http"
		}
		u.Host = r.Host
		taskUrl.Url = u.String()
		taskUrls = append(taskUrls, taskUrl)
	}

	// downloadJson, err := json.MarshalIndent(taskURLs, "", "  ")
	downloadJson, err := JsonMarshal(taskUrls)
	if err != nil {
		Error.Println(err)
	}

	//set relevant header.
	w.Header().Set("Content-Disposition",
		fmt.Sprintf("attachment; filename=%s_task_urls.json", projectName))
	_, err = io.Copy(w, bytes.NewReader(downloadJson))
	if err != nil {
		Error.Println(err)
	}
}

// handles category YAML file, sets to default values if file missing
func getCategoriesFromProjectForm(r *http.Request) []Category {
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
		Info.Printf("Miss category file and using default categories for %s.",
			labelType)

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
		Error.Println(err)
	}

	return categories
}

// handles category YAML file, sets to default values if file missing
func getAttributesFromProjectForm(r *http.Request) []Attribute {
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
		Info.Printf("Missing attribute file and"+
			"using default attributes for %s.", labelType)

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

// helper function for loading label json file to seperate indices by videoname
func handleAttributeLoad(itemPtr *Item, itemImport ItemExport,
	attributes []Attribute) {
	item := *itemPtr
	item.Attributes = map[string][]int{}
	keys := reflect.ValueOf(itemImport.Attributes).MapKeys()
	strkeys := make([]string, len(keys))
	for i := 0; i < len(keys); i++ {
		strkeys[i] = keys[i].String()
	}
	for _, key := range strkeys {
		for _, attribute := range attributes {
			if attribute.Name == key {
				for i := 0; i < len(attribute.Values); i++ {
					if itemImport.Attributes[key] == attribute.Values[i] {
						item.Attributes[key] = []int{i}
						break
					}
				}
				break
			}
		}
	}
}

// load label json file
func getItemsFromProjectForm(r *http.Request,
	attributes []Attribute) map[string][]Item {
	itemLists := make(map[string][]Item) //map[string][]Item
	var itemsImport []ItemExport
	importFile, header, err := r.FormFile("item_file")

	switch err {
	case nil:
		defer importFile.Close()

		importFileBuf := bytes.NewBuffer(nil)
		_, err = io.Copy(importFileBuf, importFile)
		if err != nil {
			Error.Println(err)
		}
		if strings.HasSuffix(header.Filename, ".json") {
			err = json.Unmarshal(importFileBuf.Bytes(), &itemsImport)
		} else {
			err = yaml.Unmarshal(importFileBuf.Bytes(), &itemsImport)
		}
		if err != nil {
			Error.Println(err)
		}

		//to seperate indexes by videoName. This also initializes indexes to 0.
		indexes := make(map[string]int)
		for _, itemImport := range itemsImport {
			item := Item{}
			item.Url = itemImport.Url
			item.Index = indexes[itemImport.VideoName]
			item.VideoName = itemImport.VideoName
			item.Timestamp = itemImport.Timestamp
			// load item attributes if needed
			if len(itemImport.Attributes) > 0 {
				handleAttributeLoad(&item, itemImport, attributes)
			}
			if len(itemImport.Labels) > 0 {
				item.LabelImport = itemImport.Labels
			}
			if itemImport.VideoName == "" {
				itemLists[" "] = append(itemLists[" "], item)
			} else {
				itemLists[itemImport.VideoName] =
					append(itemLists[itemImport.VideoName], item)
			}
			indexes[itemImport.VideoName]++
		}

	case http.ErrMissingFile:
		Error.Printf("Nothing imported")

	default:
		Error.Println(err)
	}
	return itemLists
}

func CreateTasks(project Project) {
	index := 0
	if project.Options.ItemType == "video" {
		for _, itemList := range project.Items {
			numLabelImport := 0
			numLabeledItemImport := 0
			for _, item := range itemList {
				hasLabel := false
				for _, label := range item.LabelImport {
					if label.ManualShape {
						numLabelImport++
						hasLabel = true
					}
				}
				if hasLabel {
					numLabeledItemImport++
				}
			}
			task := Task{
				ProjectOptions:       project.Options,
				Index:                index,
				Items:                itemList,
				NumFrames:            len(itemList),
				NumLabelImport:       numLabelImport,
				NumLabeledItemImport: numLabeledItemImport,
			}
			index++
			err := storage.Save(task.GetKey(), task.GetFields())
			if err != nil {
				Error.Println(err)
			}
		}

	} else {
		// otherwise, make as many tasks as required
		items := []Item{}
		for _, itemList := range project.Items {
			items = append(items, itemList...)
		}
		size := len(items)
		for i := 0; i < size; i += project.Options.TaskSize {
			numLabelImport := 0
			numLabeledItemImport := 0
			itemsSlice := items[i:Min(i+project.Options.TaskSize, size)]
			for _, item := range itemsSlice {
				numLabelImport += len(item.LabelImport)
				if item.LabelImport != nil {
					numLabeledItemImport++
				}
			}
			task := Task{
				ProjectOptions:       project.Options,
				Index:                index,
				Items:                itemsSlice,
				NumFrames:            len(itemsSlice),
				NumLabelImport:       numLabelImport,
				NumLabeledItemImport: numLabeledItemImport,
			}
			index = index + 1
			err := storage.Save(task.GetKey(), task.GetFields())
			if err != nil {
				Error.Println(err)
			}
		}
	}
	Info.Println("Created", index, "new tasks")
}

// server side create form validation
func formValidation(w http.ResponseWriter, r *http.Request) error {
	if r.FormValue("project_name") == "" {
		_, err := w.Write([]byte("Please create a project name."))
		if err != nil {
			Error.Println(err)
		}
		return errors.New("invalid form: no project name")
	}

	if r.FormValue("item_type") == "" {
		_, err := w.Write([]byte("Please choose an item type."))
		if err != nil {
			Error.Println(err)
		}
		return errors.New("invalid form: no item type")
	}

	if r.FormValue("label_type") == "" {
		_, err := w.Write([]byte("Please choose a label type."))
		if err != nil {
			Error.Println(err)
		}
		return errors.New("invalid form: no label type")
	}

	if r.FormValue("item_type") != "video" && r.FormValue("task_size") == "" {
		_, err := w.Write([]byte("Please specify a task size."))
		if err != nil {
			Error.Println(err)
		}
		return errors.New("invalid form: no task size")
	}
	// TODO: check forms are actually uploaded
	return nil
}

func gatewayHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.NotFound(w, r)
		return
	}
	gate := GatewayInfo{
		Addr: env.ModelGateHost,
		Port: env.ModelGatePort,
	}
	gateJson, err := json.Marshal(gate)
	if err != nil {
		Error.Println(err)
	}
	_, err = w.Write(gateJson)
	if err != nil {
		Error.Println(err)
	}

}

// Handles the flag value of User Management System
// during the redirection from index.html to /auth
func loadHandler(w http.ResponseWriter, r *http.Request) {
	Info.Printf("%s is requesting %s", r.RemoteAddr, r.URL)
	Info.Printf("User Management System is %s", env.UserManagement)
	// Check if WORKER_SYSTEM is On
	if env.UserManagement == "on" ||
		env.UserManagement == "On" || env.UserManagement == "ON" {
		// redirect to AWS authentication website
		authUrl := fmt.Sprintf("https://%v.auth.%v.amazoncognito.com/",
			env.DomainName, env.Region) +
			fmt.Sprintf("login?response_type=code&client_id=%v&redirect_uri=%v",
				env.ClientId, env.RedirectUri)
		http.Redirect(w, r, authUrl, 301)
	} else {
		// redirect to create
		createUrl := "/create"
		http.Redirect(w, r, createUrl, 301)
	}
}

// Handles the authenticatoin of access token
func authHandler(w http.ResponseWriter, r *http.Request) {
	// Check if WORKER_SYSTEM is On
	flag := env.UserManagement == "on" ||
		env.UserManagement == "On" || env.UserManagement == "ON"
	if !flag {
		// redirect to create
		createUrl := "/create"
		http.Redirect(w, r, createUrl, 301)
	}
	Info.Printf("%s is requesting %s", r.RemoteAddr, r.URL)

	// retrieve value from config file
	region := env.Region
	userPoolId := env.UserPoolId
	clientId := env.ClientId
	secret := env.ClientSecret
	redirectUri := env.RedirectUri
	awsTokenUrl := env.AWSTokenUrl
	awsJwkUrl := env.AwsJwkUrl
	code := r.FormValue("code")
	// check if the server received a valid authorization code,
	// if not, redirect to login page
	if code == "" {
		redirectToLogin(w, r, "Invalid authorization code")
		return
	}
	idTokenString, accessTokenString, refreshTokenString := requestToken(w, r,
		clientId, redirectUri, awsTokenUrl, code, secret)

	// Download and store the JSON Web Key (JWK) for your user pool
	jwkUrl := fmt.Sprintf(awsJwkUrl, region, userPoolId)
	jwk := getJWK(jwkUrl)
	Info.Printf("Downloading Json Web Key from Amazon")

	// veryfy accesstoken
	accessToken, err := validateAccessToken(accessTokenString,
		region, userPoolId, jwk)
	if err != nil || !accessToken.Valid {
		// failed to verify the jwt
		Error.Println(err)
		Error.Println(errors.New("Access token is not valid"))
		newUrl := "/"
		http.Redirect(w, r, newUrl, 301)
		return
	}
	Info.Printf("Access token verifed")
	// check identity by idtoken and get the user's information
	idToken, userInfo, err := validateIdToken(idTokenString,
		region, userPoolId, jwk)
	identity := userInfo.Group

	if err != nil || !idToken.Valid || identity == "" {
		// error or not valid token or empty group, redirect to index
		Error.Println(err)
		newUrl := "/"
		http.Redirect(w, r, newUrl, 301)
		return
	}
	// save refresh token in the backend
	if Users == nil {
		fmt.Printf("global variable mistake")
		return
	}
	userInfo.RefreshToken = refreshTokenString

	/* TODO: Here we are trying to track which projects are
	assigned for this user, Later the coder could feel free to use the
	'Projects' attribute of 'User' sturcture
	*/
	// load the projects information from disk for this user
	if _, ok := Users[userInfo.Id]; ok {
		userInfo.Projects = Users[userInfo.Id].Projects
	}
	Users[userInfo.Id] = &userInfo // save userinfo to memory

	// save refresh/id tokens in the cookie
	refreshTokenExpireTime := 365 * 24 * time.Hour
	// TODO: find a better expire time for the cookie,
	// maybe use the expire time of refresh token
	expiration := time.Now().Add(refreshTokenExpireTime)
	refreshTokenCookie := http.Cookie{Name: "refreshTokenScalabel",
		Value: refreshTokenString, Expires: expiration}
	idTokenCookie := http.Cookie{Name: "idScalabel",
		Value: userInfo.Id, Expires: expiration}
	http.SetCookie(w, &refreshTokenCookie)
	http.SetCookie(w, &idTokenCookie)
	if identity == "worker" {
		// if the user is not admin, redirect to user's tasks
		newUrl := "/workerDashboard"
		http.Redirect(w, r, newUrl, 301)
		return
	} else if identity == "admin" {
		// if the user is admin, redirect to admin's dashboard
		Info.Println("Admin's Cookie Saved")
		newUrl := "/adminDashboard"
		http.Redirect(w, r, newUrl, 301)
		return
	}
}

// Handles the log out action
func logOutHandler(w http.ResponseWriter, r *http.Request) {
	// get the id from cookie
	id, _ := r.Cookie("idScalabel")
	if id == nil {
		redirectToLogin(w, r, "No idCookie")
		return
	}
	// remove corresponding userInfo from backend
	Users[id.Value].RefreshToken = ""
	// reset the cookies
	refreshTokenExpireTime := 365 * 24 * time.Hour
	// TODO: find a better expire time for the cookie,
	// maybe use the expire time of refresh token
	expiration := time.Now().Add(refreshTokenExpireTime)
	refreshCookie := http.Cookie{Name: "refreshTokenScalabel",
		Value: "refreshTokenCookie", Expires: expiration}
	idCookie := http.Cookie{Name: "idScalabel", Value: "id",
		Expires: expiration}
	http.SetCookie(w, &refreshCookie)
	http.SetCookie(w, &idCookie)

	// Redirect to logOut Endpoint to log out from cognito console
	/* Replace this if you are using other authorizaition
	server instead of AWS */
	logOutUrl := fmt.Sprintf("https://%v.auth.%v.amazoncognito.com/logout",
		env.DomainName, env.Region) +
		fmt.Sprintf("?client_id=%v&logout_uri=%v", env.ClientId, env.LogOutUri)

	Info.Println(logOutUrl)
	Info.Println(env.LogOutUri, env.Region, env.ClientId)
	Info.Println("User logged out")
	http.Redirect(w, r, logOutUrl, 301)
}

// Handles the Dashboard for Worker
func workerDashboardHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.WorkerPath())
	if err != nil {
		Error.Println(err)
		http.NotFound(w, r)
		return
	}
	err = tmpl.Execute(w, "")
	if err != nil {
		Error.Println(err)
	}
}

// Handles the Dashboard for Admin
func adminDashboardHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.AdminPath())
	if err != nil {
		Error.Println(err)
		http.NotFound(w, r)
		return
	}
	err = tmpl.Execute(w, "")
	if err != nil {
		Error.Println(err)
	}
}

// Handles the posting of users information
func postUsersHandler(w http.ResponseWriter, r *http.Request) {
	// check identity:
	refreshTokenCookie, _ := r.Cookie("refreshTokenScalabel")
	idCookie, _ := r.Cookie("idScalabel")
	if refreshTokenCookie == nil {
		redirectToLogin(w, r, "No refreshTokenCookie")
		return
	} else if idCookie == nil {
		redirectToLogin(w, r, "No idCookie")
		return
	} else if !verifyRefreshToken(refreshTokenCookie.Value,
		idCookie.Value) {
		redirectToLogin(w, r, "Failed to verify: Invalid Tokens")
		return
	} else {
		group := Users[idCookie.Value].Group
		if group == "admin" { // valid to get users information
			// retrieve all the users information
			userlist := make([]User, 0, len(Users))
			for _, value := range Users {
				userlist = append(userlist, *value)
			}
			// marshal the Users as a json
			loadedUsers, err := json.Marshal(userlist)
			if err != nil {
				Error.Println(err)
			}
			// send to front end
			_, err = w.Write(loadedUsers)
			if err != nil {
				Error.Println(err)
			}
		} else {
			return
		}
	}
}

// Handles the posting of all projects' names
func postProjectNamesHandler(w http.ResponseWriter, r *http.Request) {
	// retrieve values from server's disk
	existingProjects := GetExistingProjects()
	// check if the list is empty
	if len(existingProjects) == 0 {
		existingProjects = []string{"No existing project."}
	}
	// marshal the projects' names as a json
	projectsNames, err := json.Marshal(existingProjects)
	if err != nil {
		Error.Println(err)
	}
	// send to front end
	_, err = w.Write(projectsNames)
	if err != nil {
		Error.Println(err)
	}
}

// Handles the posting of dashboard contents
func postDashboardContentsHandler(w http.ResponseWriter, r *http.Request) {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		Error.Println(err)
		return
	}
	pageData := PageData{}
	err = json.Unmarshal(body, &pageData)
	if err != nil {
		Error.Println(err)
	}
	dashboardContents, err := GetDashboardContents(pageData.Name)
	if err != nil {
		Error.Println(err)
		http.NotFound(w, r)
		return
	}
	jsonDashboardContents, err := json.Marshal(dashboardContents)
	if err != nil {
		Error.Println(err)
	} else {
		_, err = w.Write(jsonDashboardContents)
		if err != nil {
			Error.Println(err)
		}
	}
}
