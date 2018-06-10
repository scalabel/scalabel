package sat

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path"
	"strconv"
	"time"
	"unicode/utf8"
)

func GetTask(projectName string, taskIndex string) Task {
	taskPath := path.Join(env.DataDir, "tasks", projectName, taskIndex+".json")
	taskFileContents, err := ioutil.ReadFile(taskPath)
	if err != nil {
		Error.Println(err)
	}
	task := Task{}
	json.Unmarshal(taskFileContents, &task)
	return task
}

func GetTasks() []Task {
	tasksDirectoryPath := path.Join(env.DataDir, "tasks")
	tasksDirectoryContents, err := ioutil.ReadDir(tasksDirectoryPath)
	if err != nil {
		Error.Println(err)
	}
	tasks := []Task{}
	for _, projectDirectory := range tasksDirectoryContents {
		if projectDirectory.IsDir() {
			projectDirectoryPath := path.Join(env.DataDir, "tasks", projectDirectory.Name())
			projectDirectoryContents, err := ioutil.ReadDir(projectDirectoryPath)
			if err != nil {
				Error.Println(err)
			}
			for _, taskFile := range projectDirectoryContents {
				if len(taskFile.Name()) > 5 &&
					path.Ext(taskFile.Name()) == ".json" {
					taskFileContents, err := ioutil.ReadFile(
						path.Join(projectDirectoryPath, taskFile.Name()))
					if err != nil {
						Error.Println(err)
					}
					task := Task{}
					json.Unmarshal(taskFileContents, &task)
					tasks = append(tasks, task)
				}
			}
		}
	}
	return tasks
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

func GetHandlerUrl(project Project) string {
	if project.ItemType == "image" {
		if project.LabelType == "box2d" {
			return "2d_bbox_labeling"
		}
		if project.LabelType == "segmentation" {
			return "2d_seg_labeling"
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
