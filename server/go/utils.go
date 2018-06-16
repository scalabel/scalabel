package main

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path"
	"strconv"
	"time"
	"unicode/utf8"
	"strings"
)

// TODO: use actual worker ID
const DEFAULT_WORKER_ID = "default_worker"

func GetProject(projectName string) Project {
	projectsDirectoryPath := path.Join(env.ProjectsDir)
	err := os.MkdirAll(projectsDirectoryPath, 0777)
	if err != nil {
		Error.Println(err)
	}
	projectFilePath := path.Join(env.ProjectsDir, projectName, "project.json")
	projectFileContents, err := ioutil.ReadFile(projectFilePath)
	if err != nil {
		Error.Println(err)
	}
	project := Project{}
	err = json.Unmarshal(projectFileContents, &project)
	if err != nil {
		Error.Println(err)
	}
	return project
}

// DEPRECATED
func GetProjects() []Project {
	projectsDirectoryPath := path.Join(env.ProjectsDir, "projects")
	err := os.MkdirAll(projectsDirectoryPath, 0777)
	if err != nil {
		Error.Println(err)
	}
	projectsDirectoryContents, err := ioutil.ReadDir(
		projectsDirectoryPath)
	if err != nil {
		Error.Println(err)
	}
	projects := []Project{}
	for _, projectFile := range projectsDirectoryContents {
		if len(projectFile.Name()) > 5 &&
		  path.Ext(projectFile.Name()) == ".json" {
			projectFileContents, err := ioutil.ReadFile(
				path.Join(projectsDirectoryPath, projectFile.Name()))
			if err != nil {
				Error.Println(err)
			}
			project := Project{}
			err = json.Unmarshal(projectFileContents, &project)
			if err != nil {
				Error.Println(err)
			}
			projects = append(projects, project)
		}
	}
	return projects
}

func GetTask(projectName string, index string) Task {
	taskPath := path.Join(env.ProjectsDir, projectName, "tasks", index+".json")
	taskFileContents, err := ioutil.ReadFile(taskPath)
	if err != nil {
		Error.Println(err)
	}
	task := Task{}
	err = json.Unmarshal(taskFileContents, &task)
	if err != nil {
		Error.Println(err)
	}
	return task
}

func GetTasksInProject(projectName string) []Task {
	projectTasksPath := path.Join(env.ProjectsDir, projectName, "tasks")
	os.MkdirAll(projectTasksPath, 0777)
	tasksDirectoryContents, err := ioutil.ReadDir(projectTasksPath)
	if err != nil {
		Error.Println(err)
	}
	tasks := []Task{}
	for _, taskFile := range tasksDirectoryContents {
		if len(taskFile.Name()) > 5 &&
			path.Ext(taskFile.Name()) == ".json" {
			taskFileContents, err := ioutil.ReadFile(
				path.Join(projectTasksPath, taskFile.Name()))
			if err != nil {
				Error.Println(err)
			}
			task := Task{}
			err = json.Unmarshal(taskFileContents, &task)
			if err != nil {
				Error.Println(err)
			}
			tasks = append(tasks, task)
		}
	}
	return tasks
}

func GetAssignment(projectName string, taskIndex string, workerId string) Assignment {
	assignmentPath := path.Join(env.ProjectsDir, projectName, "assignments",
		taskIndex, workerId+".json")
	assignmentFileContents, err := ioutil.ReadFile(assignmentPath)
	if err != nil {
		Error.Println(err)
	}
	assignment := Assignment{}
	json.Unmarshal(assignmentFileContents, &assignment)
	return assignment
}

func CreateAssignment(projectName string, taskIndex string, workerId string) Assignment {
	task := GetTask(projectName, taskIndex)
	assignment := Assignment{
		Task: task,
		WorkerId: workerId,
		StartTime: recordTimestamp(),
	}
	assignment.Initialize()
	return assignment
}

// DEPRECATED
func GetAssignments() []Assignment {
	assignmentsDirectoryPath := path.Join(env.ProjectsDir, "assignments")
	assignmentsDirectoryContents, err := ioutil.ReadDir(
		assignmentsDirectoryPath)
	if err != nil {
		Error.Println(err)
	}
	assignments := []Assignment{}
	for _, projectDirectory := range assignmentsDirectoryContents {
		if projectDirectory.IsDir() {
			projectDirectoryPath := path.Join(env.ProjectsDir, "assignments",
				projectDirectory.Name())
			projectDirectoryContents, err := ioutil.ReadDir(
				projectDirectoryPath)
			if err != nil {
				Error.Println(err)
			}
			for _, assignmentFile := range projectDirectoryContents {
				if len(assignmentFile.Name()) > 5 &&
					path.Ext(assignmentFile.Name()) == ".json" {
					assignmentFileContents, err := ioutil.ReadFile(
						path.Join(projectDirectoryPath, assignmentFile.Name()))
					if err != nil {
						Error.Println(err)
					}
					assignment := Assignment{}
					json.Unmarshal(assignmentFileContents, &assignment)
					assignments = append(assignments, assignment)
				}
			}
		}
	}
	return assignments
}

func GetDashboardContents(projectName string) DashboardContents {
	return DashboardContents{
		Project: GetProject(projectName),
		Tasks: GetTasksInProject(projectName),
	}
}

func GetHandlerUrl(itemType string, labelType string) string {
	switch itemType {
	case "image":
		switch labelType {
		case "box2d":
			return "2d_bbox_labeling"
		case "segmentation":
			return "2d_seg_labeling"
		case "lane":
			return "2d_lane_labeling"
		default:
			return "NO_VALID_HANDLER"
		}
	case "video":
		switch labelType {
		case "box2d":
			return "video_bbox_labeling"
		default:
			return "NO_VALID_HANDLER"
		}
	}
	return "NO_VALID_HANDLER"
}

func recordTimestamp() int64 {
	// record timestamp in seconds
	return time.Now().Unix()
}

func formatTime(timestamp int64) string {
	t := time.Unix(timestamp, 0)
	return t.Format("2006-01-02_03-04-05")
}

func formatID(id int) string {
	str := strconv.Itoa(id)
	strLen := utf8.RuneCountInString(str)
	for i := 0; i < (4 - strLen); i += 1 {
		str = "0" + str
	}
	return str
}

func Exists(name string) bool {
	_, err := os.Stat(name)
	if os.IsNotExist(err) {
		return false
	} else {
		return true
	}
}

func Min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

// Behavior is similar to path stem in Python
func PathStem(name string) string {
	name = path.Base(name)
	dotIndex := strings.LastIndex(name, ".")
	if dotIndex < 0 {
		return name
	} else {
		return name[:dotIndex]
	}
}

// check duplicated project name
// return false if duplicated
func CheckProjectName(projectName string) string {
	var newName = strings.Replace(projectName, " ", "_", -1)
	os.MkdirAll(env.ProjectsDir, 0777)
	files, err := ioutil.ReadDir(env.ProjectsDir)
	if err != nil {
		return newName
	}

	for _, f := range files {
		if PathStem(f.Name()) == newName {
			Error.Printf("Project Name \"%s\" already exists.", projectName)
			return ""
		}
	}
	return newName
}

// default box2d category if category file is missing
var defaultBox2dCategories = []Category {
	{"person", nil},
	{"rider", nil},
	{"car", nil},
	{"truck", nil},
	{"bus", nil},
	{"train", nil},
	{"motor", nil},
	{"bike", nil},
	{"traffic sign", nil},
	{"traffic light", nil},
}

// default seg2d category if category file is missing
var defaultSeg2dCategories = []Category {
	{"void", []Category {
		{"unlabeled", nil},
		{"dynamic", nil},
		{"ego vehicle", nil},
		{"ground", nil},
		{"static", nil},
	}},
	{"flat", []Category {
		{"parking", nil},
		{"rail track", nil},
		{"road", nil},
		{"sidewalk", nil},
	}},
	{"construction", []Category {
		{"bridge", nil},
		{"building", nil},
		{"bus stop", nil},
		{"fence", nil},
		{"garage", nil},
		{"guard rail", nil},
		{"tunnel", nil},
		{"wall", nil},
	}},
	{"object", []Category {
		{"banner", nil},
		{"billboard", nil},
		{"fire hydrant", nil},
		{"lane divider", nil},
		{"mail box", nil},
		{"parking sign", nil},
		{"pole", nil},
		{"polegroup", nil},
		{"street light", nil},
		{"traffic cone", nil},
		{"traffic device", nil},
		{"traffic light", nil},
		{"traffic sign", nil},
		{"traffic sign frame", nil},
		{"trash can", nil},
	}},
	{"nature", []Category {
		{"terrain", nil},
		{"vegetation", nil},
	}},
	{"sky", []Category {
		{"sky", nil},
	}},
	{"human", []Category {
		{"person", nil},
		{"rider", nil},
	}},
	{"vehicle", []Category {
		{"bicycle", nil},
		{"bus", nil},
		{"car", nil},
		{"caravan", nil},
		{"motorcycle", nil},
		{"trailer", nil},
		{"train", nil},
		{"truck", nil},
	}},
}

// default lane2d category if category file is missing
var defaultLane2dCategories = []Category {
	{"road curb", nil},
	{"double white", nil},
	{"double yellow", nil},
	{"double other", nil},
	{"single white", nil},
	{"single yellow", nil},
	{"single other", nil},
	{"crosswalk", nil},
}

// default box2d attributes if attribute file is missing
var defaultBox2dAttributes = []Attribute {
	{"Occluded", "switch", "o",
		"", nil, nil, nil,
	},
	{"Truncated", "switch", "t",
		"", nil, nil, nil,
	},
	{"Traffic Light Color", "list", "", "t",
		[]string{"", "g", "y", "r"}, []string{"NA", "G", "Y", "R"},
		[]string{"white", "green", "yellow", "red"},
	},
}

// default attributes if attribute file is missing
// to avoid uncaught type error in Javascript file
var dummyAttribute = []Attribute {
	{"", "", "",
		"", nil, nil, nil,
	},
}

