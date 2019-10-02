package main

import (
	"path"
	"strconv"
)

// Functions for sat data paths (single task and worker)
// Gets key (path + filename)
func (sat *Sat) GetKey() string {
  return path.Join(sat.GetPath(), sat.getFilename())
}
// Gets path only
func (sat *Sat) GetPath() string {
  return path.Join(sat.getSubmitDir(),
    sat.User.UserId)
}
// Gets path from parameters
func GetSatPath(projectName string, taskId string,
  userId string) string {
  return path.Join(getSubmitDir(projectName, taskId),
    userId)
}
// Gets path for assignment from parameters
func GetAssignmentPath(projectName string, taskId string,
  userId string) string {
    return path.Join(getAssignmentDir(projectName, taskId),
    userId)
}

func (assignment *Assignment) GetKey() string {
	task := assignment.Task
	if assignment.SubmitTime == 0 {
		return GetAssignmentPath(task.ProjectOptions.Name,
			Index2str(task.Index), assignment.WorkerId)
	}
	return path.Join(getSubmitDir(task.ProjectOptions.Name,
		Index2str(task.Index)), assignment.WorkerId,
		strconv.FormatInt(assignment.SubmitTime, 10))
}

// Helper functions
// Gets the filename
func (sat *Sat) getFilename() string {
  return strconv.FormatInt(sat.Task.Config.SubmitTime, 10)
}
// Gets submissions directory
func (sat *Sat) getSubmitDir() string {
  return getSubmitDir(sat.Task.Config.ProjectName, sat.Task.Config.TaskId)
}
// Gets submissions directory from parameters
func getSubmitDir(projectName string, taskId string) string {
    return path.Join(projectName, "submissions", taskId)
}
// Gets assignment directory from parameters
func getAssignmentDir(projectName string, taskId string) string {
    return path.Join(projectName, "assignments", taskId)
}
