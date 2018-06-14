package main

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path"
	"strconv"
	"time"
	"unicode/utf8"
	"log"
)

func GetAssignment(projectName string, taskIndex string) Assignment {
	assignmentPath := path.Join(env.DataDir, "assignments", projectName, taskIndex+".json")
	assignmentFileContents, err := ioutil.ReadFile(assignmentPath)
	if err != nil {
		Error.Println(err)
	}
	assignment := Assignment{}
	json.Unmarshal(assignmentFileContents, &assignment)
	return assignment
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
			projectDirectoryPath := path.Join(env.DataDir, "assignments", projectDirectory.Name())
			projectDirectoryContents, err := ioutil.ReadDir(projectDirectoryPath)
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

func (task *Task) GetTaskPath() string {
	filename := strconv.Itoa(task.Index)
	dir := path.Join(
		env.DataDir,
		"tasks",
		task.ProjectName,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, filename+".json")
}

func (assignment *Assignment) GetAssignmentPath() string {
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
func checkProjName(projName string) bool {
	files, err := ioutil.ReadDir(path.Join(env.DataDir, "tasks"))
	if err != nil {
		log.Fatal(err)
	}
	for _, f := range files {
		if f.Name() == projName {
			Error.Println("Project Name - " + projName + " - already exists.")
			return false
		}
	}
	return true
}



