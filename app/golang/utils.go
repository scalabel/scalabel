package main

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
	taskPath := path.Join(env.ProjectPath,
		"data",
		"Tasks",
		projectName,
		taskIndex,
	)
	fileContents, err := ioutil.ReadFile(taskPath)
	if err != nil {
		Error.Println(err)
	}
	task := Task{}
	json.Unmarshal(fileContents, &task)
	return task
}

func GetTasks() []Task {
	dirPath := path.Join(env.ProjectPath,
		"data",
		"Tasks",
	)
	dirContents, _ := ioutil.ReadDir(dirPath)
	tasks := []Task{}
	for _, file := range dirContents {
		fileContents, err := ioutil.ReadFile(dirPath + file.Name())
		if err != nil {
			Error.Println(err)
		}
		t := Task{}
		json.Unmarshal(fileContents, &t)
		tasks = append(tasks, t)
	}
	return tasks
}

func (task *Task) GetTaskPath() string {
	filename := strconv.Itoa(task.Index)
	dir := path.Join(
		env.DataDir,
		"Tasks",
		task.ProjectName,
	)
	os.MkdirAll(dir, 0777)
	return path.Join(dir, filename+".json")
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
