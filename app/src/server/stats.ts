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
  const numPerItem = task.items.map((item) => _.size(item.labels))
  return _.sum(numPerItem)
}

/**
 * Returns the number of items with at least 1 label in the task
 */
export function countLabeledItemsTask (task: TaskType): number {
  const labeledItems = task.items.filter((item) => _.size(item.labels) > 0)
  return labeledItems.length
}

/**
 * Returns the total number of labels in the project
 */
export function countLabelsProject (tasks: TaskType[]): number {
  return _.sum(tasks.map(countLabelsTask))
}

/**
 * Returns the number of items with at least 1 label in the project
 */
export function countLabeledItemsProject (tasks: TaskType[]): number {
  return _.sum(tasks.map(countLabeledItemsTask))

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

/**
 * Get the total number of items across all tasks
 */
export function getNumItems (tasks: TaskType[]) {
  const itemsPerTask = tasks.map((task) => task.items.length)
  return _.sum(itemsPerTask)
}

/**
 * Get the number of tasks with a submission
 */
export function getNumSubmissions (tasks: TaskType[]) {
  const submittedTasks = tasks.filter((task) =>
    task.progress.submissions.length > 0)
  return submittedTasks.length
}

/**
 * Get the number of labels with each category
 */
export function getCategoryCounts (tasks: TaskType[]) {
  if (tasks.length === 0) {
    return {}
  }

  const categories = tasks[0].config.categories
  const totalCounts: { [key: string]: number } = {}
  for (const category of categories) {
    totalCounts[category] = 0
  }

  for (const task of tasks) {
    for (const item of task.items) {
      for (const label of Object.values(item.labels)) {
        for (const categoryIndex of label.category) {
          totalCounts[categories[categoryIndex]] += 1
        }
      }
    }
  }
  return totalCounts
}

/**
 * Get the number of labels with each value for each attribute
 */
export function getAttributeCounts (tasks: TaskType[]) {
  if (tasks.length === 0) {
    return {}
  }

  const attributes = tasks[0].config.attributes
  const totalCounts: { [key: string]: { [key: string]: number }} = {}
  for (const attribute of attributes) {
    const attributeCounts: { [key: string]: number } = {}
    for (const value of attribute.values) {
      attributeCounts[value] = 0
    }
    totalCounts[attribute.name] = attributeCounts
  }

  // For (const task of tasks) {
  //   for (const item of task.items) {
  //     for (const label of Object.values(item.labels)) {
  //       for (const categoryIndex of label.category) {
  //         totalCounts[categories[categoryIndex]] += 1
  //       }
  //     }
  //   }
  // }
  return totalCounts
}

interface ProjectStats {
  /** the total number of labels */
  numLabels: number
  /** the number of labeled items */
  numLabeledItems: number
  /** the total number of items */
  numItems: number
  /** the number of submitted tasks */
  numSubmittedTasks: number
  /** the total number of tasks */
  numTasks: number
  /** map from category name to count */
  categoryCounts: { [key: string]: number }
  /** map from attribute name, to counts for each value of the attribute */
  attributeCounts: { [key: string]: { [key: string]: number }}
}
/**
 * Get the stats for a collection of tasks from a project
 */
export function getProjectStats (tasks: TaskType[]): ProjectStats {
  return {
    numLabels: countLabelsProject(tasks),
    numLabeledItems: countLabeledItemsProject(tasks),
    numItems: getNumItems(tasks),
    numSubmittedTasks: getNumSubmissions(tasks),
    numTasks: tasks.length,
    categoryCounts: getCategoryCounts(tasks),
    attributeCounts: getAttributeCounts(tasks)
  }
}
