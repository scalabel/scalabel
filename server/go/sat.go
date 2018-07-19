package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/dynamodb"
	"github.com/aws/aws-sdk-go/service/dynamodb/dynamodbattribute"
	"gopkg.in/yaml.v2"
	"html/template"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"path"
	"strconv"
)

// A collection of Items to be split into Tasks. Represents one unified type
//   of annotation task with a consistent ItemType, LabelType, Category list,
//   and Attribute list. Tasks in this Project are of uniform size.
type Project struct {
	ProjectName string         //this is the primary key of project table
	Items       []Item         `json:"items" yaml"items"`
	VendorId    int            `json:"vendorId" yaml:"vendorId"`
	Options     ProjectOptions `json:"options" yaml:"options"`
}

func (project *Project) GetPath() string {
	dir := path.Join(
		env.DataDir,
		project.Options.Name,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, "project.json")
}

func (project *Project) SaveLocal() {
	path := project.GetPath()
	json, err := json.MarshalIndent(project, "", "  ")
	if err != nil {
		Error.Println(err)
	}
	err = ioutil.WriteFile(path, json, 0644)
	if err != nil {
		Error.Println(err)
	} else {
		Info.Println("Saving project file of", project.Options.Name)
	}
}

func (project *Project) SaveDatabase() {
	project.ProjectName = project.Options.Name
	av, err := dynamodbattribute.MarshalMap(project)
	input := &dynamodb.PutItemInput{
		Item:      av,
		TableName: aws.String("Project"),
	}
	_, err = svc.PutItem(input)

	if err != nil {
		Error.Println("Got error calling PutItem:")
		Error.Println(err.Error())
	}
	Info.Println("Successfully added a project to Project table on dynamodb")
}

func (project *Project) Save() {
	if env.Database {
		project.SaveDatabase()
	} else {
		project.SaveLocal()
	}
}

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
	VideoMetaData     VideoMetaData `json:"metadata" yaml:"metadata"`
}

// A workably-sized collection of Items belonging to a Project.
type Task struct {
	ProjectName    string         //primary key
	ProjectOptions ProjectOptions `json:"projectOptions" yaml:"projectOptions"`
	Index          int            `json:"index" yaml:"index"` //sort key
	Items          []Item         `json:"items" yaml:"items"`
}

func (task *Task) GetPath() string {
	dir := path.Join(
		env.DataDir,
		task.ProjectOptions.Name,
		"tasks",
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, strconv.Itoa(task.Index)+".json")
}

func (task *Task) SaveLocal() {
	path := task.GetPath()
	Info.Println(path)
	json, err := json.MarshalIndent(task, "", "  ")
	if err != nil {
		Error.Println(err)
	}
	err = ioutil.WriteFile(path, json, 0644)
	if err != nil {
		Error.Println(err)
	} else {
		Info.Println("Saving task file of", task.ProjectOptions.Name, task.Index)
	}
}

func (task *Task) SaveDatabase() {
	task.ProjectName = task.ProjectOptions.Name
	av, err := dynamodbattribute.MarshalMap(task)
	input := &dynamodb.PutItemInput{
		Item:      av,
		TableName: aws.String("Task"),
	}
	_, err = svc.PutItem(input)

	if err != nil {
		Error.Println("Got error calling PutItem:")
		Error.Println(err.Error())
	}
	Info.Println("Successfully added a task to Task table on dynamodb")
}

func (task *Task) Save() {
	if env.Database {
		task.SaveDatabase()
	} else {
		task.SaveLocal()
	}
}

// The actual assignment of a task to a worker. Contains the worker's progress.
type Assignment struct {
	PrimaryKey string //Primary Key, this is the concatenation of
	//ProjectName, TaskIndex and WorkerId
	Task            Task                   `json:"task" yaml:"task"`
	WorkerId        string                 `json:"workerId" yaml:"workerId"`
	Labels          []Label                `json:"labels" yaml:"labels"`
	Tracks          []Label                `json:"tracks" yaml:"tracks"`
	Events          []Event                `json:"events" yaml:"events"`
	StartTime       int64                  `json:"startTime" yaml:"startTime"`
	SubmitTime      int64                  `json:"submitTime" yaml:"submitTime"` //Sort Key ONLY for Submission Table
	NumLabeledItems int                    `json:"numLabeledItems" yaml:"numLabeledItems"`
	UserAgent       string                 `json:"userAgent" yaml:"userAgent"`
	IpInfo          map[string]interface{} `json:"ipInfo" yaml:"ipInfo"`
}

func (assignment *Assignment) GetAssignmentPath() string {
	dir := path.Join(
		env.DataDir,
		assignment.Task.ProjectOptions.Name,
		"assignments",
		strconv.Itoa(assignment.Task.Index),
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, assignment.WorkerId+".json")
}

func (assignment *Assignment) GetSubmissionPath() string {
	dir := path.Join(
		env.DataDir,
		assignment.Task.ProjectOptions.Name,
		"submissions",
		strconv.Itoa(assignment.Task.Index),
		assignment.WorkerId,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, strconv.FormatInt(assignment.SubmitTime, 10)+".json")
}

func (assignment *Assignment) InitializeLocal() {
	path := assignment.GetAssignmentPath()
	assignment.Serialize(path)
}

func (assignment *Assignment) SaveLocal() {
	path := assignment.GetSubmissionPath()
	assignment.Serialize(path)
}

func (assignment *Assignment) InitializeDatabase() {
	assignment.PrimaryKey = assignment.Task.ProjectOptions.Name +
		strconv.Itoa(assignment.Task.Index) +
		assignment.WorkerId
	av, err := dynamodbattribute.MarshalMap(assignment)
	input := &dynamodb.PutItemInput{
		Item:      av,
		TableName: aws.String("Assignment"),
	}
	_, err = svc.PutItem(input)

	if err != nil {
		Error.Println("Got error calling PutItem:")
		Error.Println(err.Error())
	}
	Info.Println("Successfully added a assignment to Assignment table on dynamodb")
}

func (assignment *Assignment) SaveDatabase() {
	assignment.PrimaryKey = assignment.Task.ProjectOptions.Name +
		strconv.Itoa(assignment.Task.Index) +
		assignment.WorkerId
	av, err := dynamodbattribute.MarshalMap(assignment)
	input := &dynamodb.PutItemInput{
		Item:      av,
		TableName: aws.String("Submission"),
	}
	_, err = svc.PutItem(input)

	if err != nil {
		Error.Println("Got error calling PutItem:")
		Error.Println(err.Error())
	}
	Info.Println("Successfully added a assignment to Submission table on dynamodb")

}

func (assignment *Assignment) Save() {
	if env.Database {
		assignment.SaveDatabase()
	} else {
		assignment.SaveLocal()
	}
}

func (assignment *Assignment) Initialize() {
	if env.Database {
		assignment.InitializeDatabase()
	} else {
		assignment.InitializeLocal()
	}
}

func (assignment *Assignment) Serialize(path string) {
	json, err := json.MarshalIndent(assignment, "", "  ")
	if err != nil {
		Error.Println(err)
	}
	err = ioutil.WriteFile(path, json, 0644)
	if err != nil {
		Error.Println(err)
	} else {
		Info.Println("Saving assignment file of",
			assignment.Task.ProjectOptions.Name, assignment.Task.Index,
			assignment.WorkerId, "to", path)
	}
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

type TaskURL struct {
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
	tmpl, err := template.ParseFiles(
		path.Join(env.DashboardPath()))
	if err != nil {
		Error.Println(err)
		http.NotFound(w, r)
		return
	}
	dashboardContents, err := GetDashboardContents(r.FormValue("project_name"))
	Info.Println(dashboardContents.Tasks)
	if err != nil {
		Error.Println(err)
	} else {
		Info.Println(dashboardContents.Tasks) // project is too verbose to log
		tmpl.Execute(w, dashboardContents)
	}
}

func vendorHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles(env.VendorPath())
	if err != nil {
		Error.Println(err)
		http.NotFound(w, r)
		return
	}
	dashboardContents, err := GetDashboardContents(r.FormValue("project_name"))
	if err != nil {
		Error.Println(err)
	} else {
		Info.Println(dashboardContents.Tasks) // project is too verbose to log
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
	// get frame rate from form only if this is a video
	var videoMetaData VideoMetaData
	if itemType == "video" {
		videoMetaData.TBR = r.FormValue("frame_rate")
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
	// get the task size from form
	taskSize, err := strconv.Atoi(r.FormValue("task_size"))
	if err != nil {
		Error.Println(err)
	}
	// get the vendor ID from form
	vendorId, err := strconv.Atoi(r.FormValue("vendor_id"))
	if err != nil {
		if r.FormValue("vendor_id") == "" {
			vendorId = -1
		} else {
			Error.Println(err)
		}
	}

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
		VideoMetaData:     videoMetaData,
	}
	var project = Project{
		Items:    items,
		VendorId: vendorId,
		Options:  projectOptions,
	}

	// Save project to project folder
	project.Save()

	// Initialize all the tasks
	CreateTasks(project)
}

func executeLabelingTemplateLocal(w http.ResponseWriter, r *http.Request, tmpl *template.Template) {
	// get task name from the URL
	projectName := r.URL.Query()["project_name"][0]
	taskIndex := r.URL.Query()["task_index"][0]
	if !Exists(path.Join(env.DataDir, projectName, "assignments",
		taskIndex, DEFAULT_WORKER+".json")) {
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

func executeLabelingTemplateDatabase(w http.ResponseWriter, r *http.Request, tmpl *template.Template) {
	// get task name from the URL
	projectName := r.URL.Query()["project_name"][0]
	taskIndex := r.URL.Query()["task_index"][0]
	primaryKey := projectName + taskIndex + DEFAULT_WORKER
	var assignment Assignment
	result, err := svc.GetItem(&dynamodb.GetItemInput{
		TableName: aws.String("Assignment"),
		Key: map[string]*dynamodb.AttributeValue{
			"PrimaryKey": {
				S: aws.String(primaryKey),
			},
		},
	})
	if (err != nil) || (len(result.Item) == 0) {
		assignment, err := CreateAssignment(projectName, taskIndex, DEFAULT_WORKER)
		if err != nil {
			Error.Println(err)
			return
		}
		tmpl.Execute(w, assignment)
	} else {
		err = dynamodbattribute.UnmarshalMap(result.Item, &assignment)
		if err != nil {
			Error.Println(err)
			return
		}
		tmpl.Execute(w, assignment)
	}
}

func executeLabelingTemplate(w http.ResponseWriter, r *http.Request, tmpl *template.Template) {
	if env.Database {
		executeLabelingTemplateDatabase(w, r, tmpl)
	} else {
		executeLabelingTemplateLocal(w, r, tmpl)
	}
}

// Handles the loading of an assignment given its project name, task index, and worker ID.
func postLoadAssignmentHandlerLocal(w http.ResponseWriter, r *http.Request) {
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
	if !Exists(path.Join(env.DataDir, projectName, "assignments", taskIndex,
		DEFAULT_WORKER+".json")) {
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
	loadedAssignmentJson, err := json.Marshal(loadedAssignment)
	if err != nil {
		Error.Println(err)
	}
	w.Write(loadedAssignmentJson)
}

// Handles the loading of an assignment given its project name, task index, and worker ID.
func postLoadAssignmentHandlerDatabase(w http.ResponseWriter, r *http.Request) {
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
	primaryKey := projectName + taskIndex + DEFAULT_WORKER
	result, err := svc.GetItem(&dynamodb.GetItemInput{
		TableName: aws.String("Assignment"),
		Key: map[string]*dynamodb.AttributeValue{
			"PrimaryKey": {
				S: aws.String(primaryKey),
			},
		},
	})
	if (err != nil) || (len(result.Item) == 0) {
		loadedAssignment, err = CreateAssignment(projectName, taskIndex, DEFAULT_WORKER)
	} else {
		loadedAssignment, err = GetAssignment(projectName, taskIndex,
			DEFAULT_WORKER)
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
	w.Write(loadedAssignmentJson)
}

func postLoadAssignmentHandler(w http.ResponseWriter, r *http.Request) {
	if env.Database {
		postLoadAssignmentHandlerDatabase(w, r)
	} else {
		postLoadAssignmentHandlerLocal(w, r)
	}
}

// Handles the posting of saved assignments
func postSaveHandlerLocal(w http.ResponseWriter, r *http.Request) {
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
	assignment.SubmitTime = recordTimestamp()
	// TODO: don't send all events to front end, and append these events to most recent
	submissionPath := assignment.GetSubmissionPath()
	assignmentJson, err := json.MarshalIndent(assignment, "", "  ")
	if err != nil {
		Error.Println(err)
	}
	err = ioutil.WriteFile(submissionPath, assignmentJson, 0644)
	if err != nil {
		Error.Println(err)
	} else {
		Info.Println("Saved submission file of", submissionPath)
	}
	w.Write(nil)
}

// Handles the posting of saved assignments
func postSaveHandlerDatabase(w http.ResponseWriter, r *http.Request) {
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
	assignment.SubmitTime = recordTimestamp()
	assignment.Save()
	w.Write(nil)
}

func postSaveHandler(w http.ResponseWriter, r *http.Request) {
	if env.Database {
		postSaveHandlerDatabase(w, r)
	} else {
		postSaveHandlerLocal(w, r)
	}
}

func postExportHandlerLocal(w http.ResponseWriter, r *http.Request) {
	exportFile := FileExport{}
	var projectName = r.FormValue("project_name")
	projectFilePath := path.Join(env.DataDir, projectName, "project.json")
	projectFileContents, err := ioutil.ReadFile(projectFilePath)
	if err != nil {
		Error.Println(err)
	}

	projectToLoad := Project{}
	err = json.Unmarshal(projectFileContents, &projectToLoad)
	if err != nil {
		Error.Println(err)
	}
	exportFile.Name = projectToLoad.Options.Name
	// exportFile.Categories = projectToLoad.Options.Categories
	// exportFile.Attributes = projectToLoad.Options.Attributes

	// Grab the latest submissions from all tasks
	tasks, err := GetTasksInProject(projectName)
	if err != nil {
		Error.Println(err)
		return
	}
	for _, task := range tasks {
		latestSubmission, err := GetAssignment(projectName, strconv.Itoa(task.Index), DEFAULT_WORKER)
		if err != nil {
			Error.Println(err)
			return
		}
		for _, itemToLoad := range latestSubmission.Task.Items {
			item := ItemExport{}
			item.Timestamp = 10000 // to be fixed
			item.Index = itemToLoad.Index
			for _, labelId := range itemToLoad.LabelIds {
				var labelToLoad Label
				for _, label := range latestSubmission.Labels {
					if label.Id == labelId {
						labelToLoad = label
						break
					}
				}
				label := LabelExport{}
				label.Id = labelId
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
				item.Labels = append(item.Labels, label)
			}
			exportFile.Items = append(exportFile.Items, item)
		}
	}

	exportJson, err := json.MarshalIndent(exportFile, "", "  ")
	if err != nil {
		Error.Println(err)
	}

	//set relevant header.
	w.Header().Set("Content-Disposition", "attachment; filename="+projectName+"_Results.json")
	io.Copy(w, bytes.NewReader(exportJson))
}

// Handles the export of submitted assignments
func postExportHandlerDatabase(w http.ResponseWriter, r *http.Request) {
	exportFile := FileExport{}
	var projectName = r.FormValue("project_name")
	result, err := svc.GetItem(&dynamodb.GetItemInput{
		TableName: aws.String("Project"),
		Key: map[string]*dynamodb.AttributeValue{
			"ProjectName": {
				S: aws.String(projectName),
			},
		},
	})
	projectToLoad := Project{}
	err = dynamodbattribute.UnmarshalMap(result.Item, &projectToLoad)
	if err != nil {
		Error.Println(err)
	}
	exportFile.Name = projectToLoad.Options.Name
	// exportFile.Categories = projectToLoad.Options.Categories
	// exportFile.Attributes = projectToLoad.Options.Attributes
	// Grab the latest submissions from all tasks
	tasks, err := GetTasksInProject(projectName)
	if err != nil {
		Error.Println(err)
		return
	}
	for _, task := range tasks {
		latestSubmission, err := GetAssignment(projectName, strconv.Itoa(task.Index), DEFAULT_WORKER)
		if err != nil {
			Error.Println(err)
			return
		}
		for _, itemToLoad := range latestSubmission.Task.Items {
			item := ItemExport{}
			item.Timestamp = 10000 // to be fixed
			item.Index = itemToLoad.Index
			for _, labelId := range itemToLoad.LabelIds {
				var labelToLoad Label
				for _, label := range latestSubmission.Labels {
					if label.Id == labelId {
						labelToLoad = label
						break
					}
				}
				label := LabelExport{}
				label.Id = labelId
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
				item.Labels = append(item.Labels, label)
			}
			exportFile.Items = append(exportFile.Items, item)
		}
	}

	exportJson, err := json.MarshalIndent(exportFile, "", "  ")
	if err != nil {
		Error.Println(err)
	}

	//set relevant header.
	w.Header().Set("Content-Disposition", "attachment; filename="+projectName+"_Results.json")
	io.Copy(w, bytes.NewReader(exportJson))
}

func postExportHandler(w http.ResponseWriter, r *http.Request) {
	if env.Database {
		postExportHandlerDatabase(w, r)
	} else {
		postExportHandlerLocal(w, r)
	}
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
		task.Save()
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
			task.Save()
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
