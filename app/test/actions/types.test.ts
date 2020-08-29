import {
  addLabel,
  addTrack,
  initSessionAction,
  linkLabels,
  splitPane,
  submit,
  updateTask
} from "../../src/action/common"
import { isTaskAction } from "../../src/const/action"
import { makeLabel, makeTask } from "../../src/functional/states"
import { BaseAction } from "../../src/types/action"
import { SplitType } from "../../src/types/state"

test("Test task action checker", () => {
  // Test some subset of task actions
  const taskActions: BaseAction[] = [
    addLabel(0, makeLabel()),
    linkLabels(0, []),
    addTrack([], "", [], []),
    submit()
  ]
  for (const taskAction of taskActions) {
    expect(isTaskAction(taskAction)).toBe(true)
  }

  // Test some subset of non-task actions
  const notTaskActions: BaseAction[] = [
    updateTask(makeTask()),
    initSessionAction(),
    splitPane(0, SplitType.HORIZONTAL, 0)
  ]

  for (const notTaskAction of notTaskActions) {
    expect(isTaskAction(notTaskAction)).toBe(false)
  }
})
