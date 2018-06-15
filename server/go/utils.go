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

func GetProject(projectName string) Project {
	projectsDirectoryPath := path.Join(env.DataDir, "projects")
	err := os.MkdirAll(projectsDirectoryPath, 0777)
	if err != nil {
		Error.Println(err)
	}
	projectFilePath := path.Join(env.DataDir, "projects", projectName+".json")
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

func GetProjects() []Project {
	projectsDirectoryPath := path.Join(env.DataDir, "projects")
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

func GetAssignment(projectName string, taskIndex string) Assignment {
	assignmentPath := path.Join(env.DataDir, "assignments", projectName,
		taskIndex+".json")
	assignmentFileContents, err := ioutil.ReadFile(assignmentPath)
	if err != nil {
		Error.Println(err)
	}
	assignment := Assignment{}
	json.Unmarshal(assignmentFileContents, &assignment)
	return assignment
}

func GetAssignmentsInProject(projectName string) []Assignment {
	projectAssignmentsPath := path.Join(env.DataDir, "assignments",
		projectName)
	assignmentsDirectoryContents, err := ioutil.ReadDir(projectAssignmentsPath)
	if err != nil {
		Error.Println(err)
	}
	assignments := []Assignment{}
	for _, assignmentFile := range assignmentsDirectoryContents {
		if len(assignmentFile.Name()) > 5 &&
			path.Ext(assignmentFile.Name()) == ".json" {
			assignmentFileContents, err := ioutil.ReadFile(
				path.Join(projectAssignmentsPath, assignmentFile.Name()))
			if err != nil {
				Error.Println(err)
			}
			assignment := Assignment{}
			json.Unmarshal(assignmentFileContents, &assignment)
			assignments = append(assignments, assignment)
		}
	}
	return assignments
}

func GetAssignments() []Assignment {
	assignmentsDirectoryPath := path.Join(env.DataDir, "assignments")
	assignmentsDirectoryContents, err := ioutil.ReadDir(
		assignmentsDirectoryPath)
	if err != nil {
		Error.Println(err)
	}
	assignments := []Assignment{}
	for _, projectDirectory := range assignmentsDirectoryContents {
		if projectDirectory.IsDir() {
			projectDirectoryPath := path.Join(env.DataDir, "assignments",
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
		Assignments: GetAssignmentsInProject(projectName),
	}
}

func (project *Project) GetPath() string {
	return path.Join(
		env.DataDir,
		"projects",
		project.Name+".json",
	)
}

func (task *Task) GetPath() string {
	filename := strconv.Itoa(task.Index)
	dir := path.Join(
		env.DataDir,
		"tasks",
		task.ProjectName,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, filename+".json")
}

func (assignment *Assignment) GetPath() string {
	filename := strconv.Itoa(assignment.Task.Index)
	dir := path.Join(
		env.DataDir,
		"assignments",
		assignment.Task.ProjectName,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, filename+".json")
}

func GetHandlerUrl(project Project) string {
	if project.ItemType == "image" {
		if project.LabelType == "box2d" {
			return "2d_bbox_labeling"
		}
		if project.LabelType == "segmentation" {
			return "2d_seg_labeling"
		}
		if project.LabelType == "lane" {
        	return "2d_lane_labeling"
        }
	}
	if project.ItemType == "video" {
		if project.LabelType == "box2d" {
			return "video_bbox_labeling"
		}
	}
	// if project.ItemType == "pointcloud" {
	// 	if project.LabelType == "box3d" {
	// 		return "" // ???
	// 	}
	// }
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

// check duplicated project name
// return false if duplicated
func checkProjectName(projectName string) string {
	var newName = strings.Replace(projectName, " ", "_", -1)
	dir := path.Join(env.DataDir, "tasks")
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return newName
	}

	for _, f := range files {
		if f.Name() == newName {
			Error.Println("Project Name - " + projectName + " - already exists.")
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

