package main

import (
	"path"
  "strconv"
	"testing"
)

func assertEqual(p1, p2 string, t *testing.T) {
  if p1 != p2 {
    t.Fatalf("Wrong path: %s != %s", p1, p2)
  }
}

// Tests that paths are correct
func TestPaths(t *testing.T) {
  sat, _, err := ReadSampleSatData()
  if err != nil {
    t.Fatal(err)
  }

  projectName := sat.Task.Config.ProjectName
  userId := sat.User.UserId
  taskId := sat.Task.Config.TaskId
  submitTime := strconv.FormatInt(sat.Task.Config.SubmitTime, 10)

  // Test that the sat submission paths are correct
  satKey := path.Join(projectName, "submissions", taskId, userId, submitTime)
  assertEqual(sat.GetKey(), satKey, t)
  satPath := path.Join(projectName, "submissions", taskId, userId)
  assertEqual(sat.GetPath(), satPath, t)
  assertEqual(GetSatPath(projectName, taskId, userId), satPath, t)

  // Test that the sat assignment path is correct
  satAssignment := path.Join(projectName, "assignments", taskId, userId)
  assertEqual(satAssignment, GetAssignmentPath(projectName, taskId, userId), t)
}
