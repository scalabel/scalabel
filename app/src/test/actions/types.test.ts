import {
  addLabel, addTrack, initSessionAction,
  linkLabels, splitPane, submit, updateTask } from '../../js/action/common'
import { BaseAction, isTaskAction } from '../../js/action/types'
import { makeLabel, makeTask } from '../../js/functional/states'
import { SplitType } from '../../js/functional/types'

test('Test task action checker', () => {
  // Test some subset of task actions
  const taskActions: BaseAction[] = [
    addLabel(0, makeLabel()),
    linkLabels(0, []),
    addTrack([], '', [], [], []),
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
