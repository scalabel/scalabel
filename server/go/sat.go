package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"github.com/mitchellh/mapstructure"
	"gopkg.in/yaml.v2"
	"html/template"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"path"
	"strconv"
)

type Serializable interface {
	GetKey() string
	GetFields() map[string]interface{}
}

//implements Serializable
type Project struct {
	Items    []Item         `json:"items" yaml"items"`
	VendorId int            `json:"vendorId" yaml:"vendorId"`
	Options  ProjectOptions `json:"options" yaml:"options"`
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
	ProjectOptions ProjectOptions `json:"projectOptions" yaml:"projectOptions"`
	Index          int            `json:"index" yaml:"index"`
	Items          []Item         `json:"items" yaml:"items"`
}

func (task *Task) GetKey() string {
	return path.Join(task.ProjectOptions.Name, "tasks", strconv.Itoa(task.Index))
}

func (task *Task) GetFields() map[string]interface{} {
	return map[string]interface{}{
		"ProjectOptions": task.ProjectOptions,
		"Index":          task.Index,
		"Items":          task.Items,
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

func (assignment *Assignment) GetKey() string {
	task := assignment.Task
	if assignment.SubmitTime == 0 {
		return path.Join(task.ProjectOptions.Name, "assignments", strconv.Itoa(task.Index),
			assignment.WorkerId)
	} else {
		return path.Join(task.ProjectOptions.Name, "submissions", strconv.Itoa(task.Index),
			assignment.WorkerId, strconv.FormatInt(assignment.SubmitTime, 10))
	}
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
	TaskSize          int           `json:"taskSize" yaml:"taskSize"`
	HandlerUrl        string        `json:"handlerUrl" yaml:"handlerUrl"`
	PageTitle         string        `json:"pageTitle" yaml:"pageTitle"`
	Categories        []Category    `json:"categories" yaml:"categories"`
	NumLeafCategories int           `json:"numLeafCategories" yaml:"numLeafCategories"`
	Attributes        []Attribute   `json:"attributes" yaml:"attributes"`
    LabelImport       []ItemExport  `json:"labelImport" yaml:"labelImport"`
	Instructions      string        `json:"instructions" yaml:"instructions"`
	DemoMode          bool          `json:"demoMode" yaml:"demoMode"`
	VideoMetaData     VideoMetaData `json:"videoMetaData" yaml:"videoMetaData"`
	InterpolationMode string        `json:"interpolationMode" yaml:"interpolationMode"`
	Detections        []Detection   `json:"detections" yaml:"detections"`
}

// An item is something to be annotated e.g. Image, PointCloud
type Item struct {
	Url         string                 `json:"url" yaml:"url"`
	Index       int                    `json:"index" yaml:"index"`
	LabelIds    []int                  `json:"labelIds" yaml:"labelIds"`
	GroundTruth []Label                `json:"groundTruth" yaml:"groundTruth"`
	Data        map[string]interface{} `json:"data" yaml:"data"`
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
	Project Project `json:"project" yaml:"project"`
	Tasks   []Task  `json:"tasks" yaml:"tasks"`
}

type TaskURL struct { //shared type
	URL string `json:"url" yaml:"url"`
}

// unescaped marshal used to encode url string
func JSONMarshal(t interface{}) ([]byte, error) {
	buffer := &bytes.Buffer{}
	encoder := json.NewEncoder(buffer)
	encoder.SetEscapeHTML(false)
	err := encoder.Encode(t)
	return buffer.Bytes(), err
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

func countCategories(categories []Category) int {
	count := 0
	for _, category := range categories {
		if len(category.Subcategories) > 0 {
			count += countCategories(category.Subcategories)
		} else {
			count += 1
		}
	}
	return count
}

func dashboardHandler(w http.ResponseWriter, r *http.Request) {
	// use template to insert assignment links
	funcMap := template.FuncMap{"countLabeledImage": countLabeledImage,
		"countLabelInTask": countLabelInTask}
	tmpl, err := template.New("dashboard.html").Funcs(funcMap).ParseFiles(
		path.Join(env.DashboardPath()))
	if err != nil {
		Error.Println(err)
		http.NotFound(w, r)
		return
	}
	dashboardContents, err := GetDashboardContents(r.FormValue("project_name"))
	if err != nil {
		Error.Println(err)
	} else {
		//Info.Println(dashboardContents.Tasks) // project is too verbose to log
		tmpl.Execute(w, dashboardContents)
	}
}

func vendorHandler(w http.ResponseWriter, r *http.Request) {
	funcMap := template.FuncMap{"countLabeledImage": countLabeledImage,
		"countLabelInTask": countLabelInTask}
	tmpl, err := template.New("vendor.html").Funcs(funcMap).ParseFiles(env.VendorPath())
	if err != nil {
		Error.Println(err)
		http.NotFound(w, r)
		return
	}
	dashboardContents, err := GetDashboardContents(r.FormValue("project_name"))
	if err != nil {
		Error.Println(err)
	} else {
		// Info.Println(dashboardContents.Tasks) // project is too verbose to log
		tmpl.Execute(w, dashboardContents)
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

	// make sure the project name in the form is new
	var projectName = CheckProjectName(r.FormValue("project_name"))
	if projectName == "" {
		w.Write([]byte("Project Name already exists."))
		return
	}
	// get item type from form
	itemType := r.FormValue("item_type")
	// get frame rate and interpolation mode from form only if this is a video
	var videoMetaData VideoMetaData
	interpolationMode := "linear"
	var detections []Detection
	if itemType == "video" {
		videoMetaData.TBR = r.FormValue("frame_rate")
		interpolationMode = r.FormValue("interpolation_mode")
	}
	// get label type from form
	labelType := r.FormValue("label_type")
	// get page title from form
	pageTitle := r.FormValue("page_title")
	// parse the item list YML from form
	items := getItemsFromProjectForm(w, r)
	if itemType == "video" {
		videoMetaData.NumFrames = strconv.Itoa(len(items))
	}
	// parse the category list YML from form
	categories := getCategoriesFromProjectForm(r)
	numLeafCategories := countCategories(categories)
	// parse the attribute list YML from form
	attributes := getAttributesFromProjectForm(r)
	// get the imported labels from form
	labelImport := getImportFromProjectForm(r)
	// get the task size from form
	var taskSize int
	if itemType != "video" {
		ts, err := strconv.Atoi(r.FormValue("task_size"))
		if err != nil {
			Error.Println(err)
			return
		}
		taskSize = ts
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

	demoMode := r.FormValue("demo_mode") == "on"

	// This prefix determines which handler will deal with labeling sessions
	//   for this project. Uniquely determined by item type and label type.
	handlerUrl := GetHandlerUrl(itemType, labelType)

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
		LabelImport:       labelImport,
		Instructions:      instructions,
		DemoMode:          demoMode,
		VideoMetaData:     videoMetaData,
		InterpolationMode: interpolationMode,
		Detections:        detections,
	}
	var project = Project{
		Items:    items,
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

func executeLabelingTemplate(w http.ResponseWriter, r *http.Request, tmpl *template.Template) {
	// get task name from the URL
	projectName := r.URL.Query()["project_name"][0]
	taskIndex := r.URL.Query()["task_index"][0]
	if !storage.HasKey(path.Join(projectName, "assignments",
		taskIndex, DEFAULT_WORKER)) {
		// if assignment does not exist, create it
		assignment, err := CreateAssignment(projectName, taskIndex, DEFAULT_WORKER)
		if err != nil {
			Error.Println(err)
			return
		}
		tmpl.Execute(w, assignment)
	} else {
		// otherwise, get that assignment
		assignment, err := GetAssignment(projectName, taskIndex, DEFAULT_WORKER)
		if err != nil {
			Error.Println(err)
			return
		}
		tmpl.Execute(w, assignment)
	}
}

// Handles the loading of an assignment given its project name, task index, and worker ID.
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
	taskIndex := strconv.Itoa(assignmentToLoad.Task.Index)
	var loadedAssignment Assignment
	if !storage.HasKey(path.Join(projectName, "assignments",
		taskIndex, DEFAULT_WORKER)) {
		// if assignment does not exist, create it
		// TODO: resolve tension between this function and executeLabelingTemplate()
		loadedAssignment, err = CreateAssignment(projectName, taskIndex,
			DEFAULT_WORKER)
		if err != nil {
			Error.Println(err)
			return
		}
	} else {
		loadedAssignment, err = GetAssignment(projectName, taskIndex,
			DEFAULT_WORKER)
		if err != nil {
			Error.Println(err)
			return
		}
		loadedAssignment.StartTime = recordTimestamp()
	}
	Error.Println(loadedAssignment)
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
	var fields map[string]interface{}
	err = json.Unmarshal(body, &fields)
	if err != nil {
		Error.Println(err)
	}
	fields["SubmitTime"] = recordTimestamp()
	assignment := Assignment{}
	mapstructure.Decode(fields, &assignment)
	if assignment.Task.ProjectOptions.DemoMode {
		Error.Println(errors.New("Can't save a demo project."))
		w.Write(nil)
		return
	}
	// TODO: don't send all events to front end, and append these events to most recent
	err = storage.Save(assignment.GetKey(), assignment.GetFields())
	if err != nil {
		Error.Println(err)
	}
	w.Write(nil)
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
	mapstructure.Decode(fields, &projectToLoad)

	// Grab the latest submissions from all tasks
	tasks, err := GetTasksInProject(projectName)
	if err != nil {
		Error.Println(err)
		return
	}
	items := []ItemExport{}
	for _, task := range tasks {
		latestSubmission, err := GetAssignment(projectName, strconv.Itoa(task.Index), DEFAULT_WORKER)
		if err != nil {
			Error.Println(err)
		}
		for _, itemToLoad := range latestSubmission.Task.Items {
			item := ItemExport{}
			if (projectToLoad.Options.ItemType == "video") {
                item.VideoName = projectToLoad.Options.Name + "_" + strconv.Itoa(task.Index)
                item.Index = itemToLoad.Index
            }
			item.Timestamp = 10000 // to be fixed
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
					label.Seg2d = ParseSeg2d(labelToLoad.Data)
				case "lane":
					label.Seg2d = ParseSeg2d(labelToLoad.Data)
				}
				label.Manual = true
				if (projectToLoad.Options.ItemType == "video") {
                    label.Manual = labelToLoad.Keyframe
                    label.Id = labelToLoad.ParentId
                } else {
                    label.Manual = true
                    label.Id = labelId
                }
				item.Labels = append(item.Labels, label)
			}
			items = append(items, item)
		}
	}

	exportJson, err := json.MarshalIndent(items, "", "  ")
	if err != nil {
		Error.Println(err)
	}

	//set relevant header.
	w.Header().Set("Content-Disposition", "attachment; filename="+projectName+"_Results.json")
	io.Copy(w, bytes.NewReader(exportJson))
}

// Handles the download of submitted assignments
func downloadTaskURLHandler(w http.ResponseWriter, r *http.Request) {
	var projectName = r.FormValue("project_name")
	tasks, err := GetTasksInProject(projectName)
	if err != nil {
		Error.Println(err)
		return
	}

	taskURLs := []TaskURL{}
	for _, task := range tasks {
		taskURL := TaskURL{}
		u, err := url.Parse(task.ProjectOptions.HandlerUrl)
		if err != nil {
			log.Fatal(err)
		}
		q := u.Query()
		q.Set("project_name", projectName)
		q.Set("task_index", strconv.Itoa(task.Index))
		u.RawQuery = q.Encode()
		if r.TLS != nil {
			u.Scheme = "https"
		} else {
			u.Scheme = "http"
		}
		u.Host = r.Host
		taskURL.URL = u.String()
		taskURLs = append(taskURLs, taskURL)
	}

	// downloadJson, err := json.MarshalIndent(taskURLs, "", "  ")
	downloadJson, err := JSONMarshal(taskURLs)
	if err != nil {
		Error.Println(err)
	}

	//set relevant header.
	w.Header().Set("Content-Disposition", "attachment; filename="+projectName+"_TaskURLs.json")
	io.Copy(w, bytes.NewReader(downloadJson))
}

// handles item YAML file
func getItemsFromProjectForm(w http.ResponseWriter, r *http.Request) []Item {
	var items []Item
	itemFile, _, err := r.FormFile("item_file")

	switch err {
	case nil:
		defer itemFile.Close()

		itemFileBuf := bytes.NewBuffer(nil)
		_, err = io.Copy(itemFileBuf, itemFile)
		if err != nil {
			Error.Println(err)
		}
		err = yaml.Unmarshal(itemFileBuf.Bytes(), &items)
		if err != nil {
			Error.Println(err)
		}
		// set the indices properly for each item
		for i := 0; i < len(items); i++ {
			items[i].Index = i
		}
	case http.ErrMissingFile:
		w.Write([]byte("Please upload an item file."))
		Error.Println(err)

	default:
		Error.Println(err)
	}
	return items
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

// load label json file
func getImportFromProjectForm(r *http.Request) []ItemExport {
	var labelImport []ItemExport
	importFile, _, err := r.FormFile("label_import")

	switch err {
	case nil:
		defer importFile.Close()

		importFileBuf := bytes.NewBuffer(nil)
		_, err = io.Copy(importFileBuf, importFile)
		if err != nil {
			Error.Println(err)
		}
		err = json.Unmarshal(importFileBuf.Bytes(), &labelImport)
		if err != nil {
			Error.Println(err)
		}

	case http.ErrMissingFile:
		Info.Printf("Nothing imported")

	default:
		log.Println(err)
	}

	return labelImport
}

func CreateTasks(project Project) {
	index := 0
	if project.Options.ItemType == "video" {
		// if the project is on video, only make 1 task
		task := Task{
			ProjectOptions: project.Options,
			Index:          0,
			Items:          project.Items,
		}
		index = 1
		err := storage.Save(task.GetKey(), task.GetFields())
		if err != nil {
			Error.Println(err)
		}
	} else {
		// otherwise, make as many tasks as required
		size := len(project.Items)
		for i := 0; i < size; i += project.Options.TaskSize {
			task := Task{
				ProjectOptions: project.Options,
				Index:          index,
				Items:          project.Items[i:Min(i+project.Options.TaskSize, size)],
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
		w.Write([]byte("Please create a project name."))
		return errors.New("Invalid form: no project name.")
	}

	if r.FormValue("item_type") == "" {
		w.Write([]byte("Please choose an item type."))
		return errors.New("Invalid form: no item type.")
	}

	if r.FormValue("label_type") == "" {
		w.Write([]byte("Please choose a label type."))
		return errors.New("Invalid form: no label type.")
	}

	if r.FormValue("item_type") != "video" && r.FormValue("task_size") == "" {
		w.Write([]byte("Please specify a task size."))
		return errors.New("Invalid form: no task size.")
	}
	// TODO: check forms are actually uploaded
	return nil
}
