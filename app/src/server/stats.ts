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
 * Returns [numLabeledItems, numLabels]
 * numLabeledItems is the number of items with at least 1 label in the task
 * numLabels is the total number of labels in the task
 */
export function countLabels (task: TaskType): [number, number] {
  let numLabeledItems = 0
  let numLabels = 0
  for (const item of task.items) {
    const currNumLabels = Object.keys(item.labels).length
    if (item.labels && currNumLabels > 0) {
      numLabeledItems++
      numLabels += currNumLabels
    }
  }
  return [numLabeledItems, numLabels]
}

/**
 * Extracts TaskOptions from a Task
 */
export function getTaskOptions (task: TaskType): TaskOptions {
  const [numLabeledItems, numLabels] = countLabels(task)
  return {
    numLabeledItems: numLabeledItems.toString(),
    numLabels: numLabels.toString(),
    submissions: task.progress.submissions,
    handlerUrl: task.config.handlerUrl
  }
}
