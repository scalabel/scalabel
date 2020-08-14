import _ from 'lodash'
import { ProjectOptions, TaskOptions } from '../components/dashboard'
import { Project } from '../types/project'
import { TaskType } from '../types/state'

/**
 * Extract ProjectOption from a Project
 */
export function getProjectOptions (project: Project): ProjectOptions {
  return {
    name: project.config.projectName,
    itemType: project.config.itemType,
    labelTypes: project.config.labelTypes,
    taskSize: project.config.taskSize,
    numItems: project.items.length,
    numLeafCategories: project.config.categories.length,
    numAttributes: project.config.attributes.length
  }
}

/**
 * Returns the total number of labels in the task
 */
export function countLabelsTask (task: TaskType): number {
  const numPerItem = _.map(task.items, (item) => _.size(item.labels))
  return _.sum(numPerItem)
}

/**
 * Returns the number of items with at least 1 label in the task
 */
export function countLabeledItemsTask (task: TaskType): number {
  const numPerItem = _.map(task.items, (item) => _.size(item.labels))
  return _.filter(numPerItem, (x) => x > 0).length
}

/**
 * Returns the total number of labels in the project
 */
export function countLabelsProject (tasks: TaskType[]): number {
  return _.sum(_.map(tasks, countLabelsTask))
}

/**
 * Returns the number of items with at least 1 label in the project
 */
export function countLabeledItemsProject (tasks: TaskType[]): number {
  return _.sum(_.map(tasks, countLabeledItemsTask))

}

/**
 * Extracts TaskOptions from a Task
 */
export function getTaskOptions (task: TaskType): TaskOptions {
  const numLabeledItems = countLabeledItemsTask(task)
  const numLabels = countLabelsTask(task)
  return {
    numLabeledItems: numLabeledItems.toString(),
    numLabels: numLabels.toString(),
    submissions: task.progress.submissions,
    handlerUrl: task.config.handlerUrl
  }
}
