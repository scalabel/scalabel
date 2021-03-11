import {
  addLabel,
  addLabelsToItem,
  addTrack,
  changeLabelsProps,
  changeShapesInItems,
  deleteLabel,
  makeNullAction,
  makeSequential
} from "../action/common"
import { deleteTracks, terminateTracks } from "../action/track"
import Session, { dispatch, getState } from "../common/session"
import { Track } from "../common/track"
import {
  ActionType,
  AddLabelsAction,
  AddTrackAction,
  BaseAction,
  DeleteLabelsAction
} from "../types/action"
import { LabelIdMap, ShapeIdMap } from "../types/state"
import Label2D from "./2d/label2d"
import Label3D from "./3d/label3d"

interface ItemLabelIdMap {
  [index: number]: LabelIdMap
}
interface ItemShapeIdMap {
  [index: number]: ShapeIdMap
}

/**
 * Commit changed shapes to state
 *
 * @param {ItemShapeIdMap} updatedShapes
 */
function commitShapesToState(updatedShapes: ItemShapeIdMap): ActionType {
  if (Object.keys(updatedShapes).length === 0) {
    return makeNullAction("no shape to commit")
  }
  // Update existing shapes
  const itemIndices = Object.keys(updatedShapes).map((index) => Number(index))
  const shapeIds = []
  const shapes = []
  for (const index of itemIndices) {
    const itemShapeIds = []
    const indexShapes = []
    for (const shapeId of Object.keys(updatedShapes[index])) {
      itemShapeIds.push(shapeId)
      indexShapes.push(updatedShapes[index][shapeId])
    }
    shapeIds.push(itemShapeIds)
    shapes.push(indexShapes)
  }
  return changeShapesInItems(itemIndices, shapeIds, shapes)
}

/**
 * Commit changed labels to state
 *
 * @param {ItemLabelIdMap} updatedLabels
 */
function commitLabelsToState(updatedLabels: ItemLabelIdMap): ActionType {
  if (Object.keys(updatedLabels).length === 0) {
    return makeNullAction("no label to commit")
  }
  // Update existing labels
  const itemIndices = Object.keys(updatedLabels).map((index) => Number(index))
  const labelIds = []
  const labels = []
  for (const index of itemIndices) {
    const itemLabelIds = []
    const indexLabels = []
    for (const labelId of Object.keys(updatedLabels[index])) {
      itemLabelIds.push(labelId)
      indexLabels.push(updatedLabels[index][labelId])
    }
    labelIds.push(itemLabelIds)
    labels.push(indexLabels)
  }
  return changeLabelsProps(itemIndices, labelIds, labels)
}

/**
 * Update track from a label
 *
 * @param {(Readonly<Label2D> | Readonly<Label3D>)} drawable
 * @param {ItemLabelIdMap} updatedLabels
 * @param {ItemShapeIdMap} updatedShapes
 */
function updateTrack(
  drawable: Readonly<Label2D> | Readonly<Label3D>,
  updatedLabels: ItemLabelIdMap,
  updatedShapes: ItemShapeIdMap
): void {
  if (!(drawable.trackId in Session.tracks)) {
    return
  }
  const track = Session.tracks[drawable.trackId]
  track.update(drawable.label.item, drawable)

  track.updatedIndices.forEach((index) => {
    if (!(index in updatedShapes)) {
      updatedShapes[index] = {}
    }
    const shapes = track.getShapes(index)
    for (const shape of shapes) {
      updatedShapes[index][shape.id] = shape
    }

    if (!(index in updatedLabels)) {
      updatedLabels[index] = {}
    }
    const label = track.getLabel(index)
    if (label !== null) {
      updatedLabels[index][label.id] = label
    }
  })
  track.clearUpdatedIndices()
}

/**
 * Update a single label
 *
 * @param {(Readonly<Label2D> | Readonly<Label3D>)} drawable
 * @param {ItemLabelIdMap} updatedLabels
 * @param {ItemShapeIdMap} updatedShapes
 */
function updateLabel(
  drawable: Readonly<Label2D> | Readonly<Label3D>,
  updatedLabels: ItemLabelIdMap,
  updatedShapes: ItemShapeIdMap
): void {
  const shapes = drawable.shapes()
  if (!(drawable.item in updatedShapes)) {
    updatedShapes[drawable.item] = {}
  }
  for (const shape of shapes) {
    updatedShapes[drawable.item][shape.id] = shape
  }

  if (!(drawable.item in updatedLabels)) {
    updatedLabels[drawable.item] = {}
  }
  updatedLabels[drawable.item][drawable.labelId] = drawable.label
}

/**
 * Add new tracks from a newly added label
 *
 * @param {(Readonly<Label2D> | Readonly<Label3D>)} drawable
 * @param {number} numItems
 */
function addNewTrack(
  drawable: Readonly<Label2D> | Readonly<Label3D>,
  numItems: number
): AddTrackAction {
  const track = new Track()
  let parentTrack
  if (drawable.parent !== null && drawable.parent.trackId in Session.tracks) {
    parentTrack = Session.tracks[drawable.parent.trackId]
  }
  track.init(drawable.item, drawable, numItems - drawable.item + 1, parentTrack)
  const indices = []
  const labels = []
  const types = []
  const shapes = []
  for (let i = 0; i < numItems; i++) {
    const label = track.getLabel(i)
    if (label !== null) {
      const indexedShapes = track.getShapes(i)
      indices.push(i)
      labels.push(label)
      types.push(indexedShapes.map((s) => s.shapeType))
      shapes.push(indexedShapes.map((s) => s))
    }
  }
  return addTrack(indices, track.type, labels, shapes)
}

/**
 * Add a new label from a drawable
 *
 * @param {(Readonly<Label2D> | Readonly<Label3D>)} drawable
 */
function addNewLabel(
  drawable: Readonly<Label2D> | Readonly<Label3D>
): AddLabelsAction {
  return addLabel(drawable.item, drawable.label, drawable.shapes())
}

/**
 * Terminate track from current drawable
 *
 * @param {(Readonly<Label2D> | Readonly<Label3D>)} drawable
 * @param {number} numItems
 */
function terminateTrackFromDrawable(
  drawable: Readonly<Label2D> | Readonly<Label3D>,
  numItems: number
): DeleteLabelsAction {
  const track = getState().task.tracks[drawable.label.track]
  if (drawable.item === 0) {
    return deleteTracks([track])
  } else {
    return terminateTracks([track], drawable.item, numItems)
  }
}

/**
 * Delete an invalid existing label
 *
 * @param {(Readonly<Label2D> | Readonly<Label3D>)} drawable
 */
function deleteInvalidLabel(
  drawable: Readonly<Label2D> | Readonly<Label3D>
): DeleteLabelsAction {
  return deleteLabel(drawable.item, drawable.labelId)
}

/**
 * Commit 2D labels to state
 *
 * @param updatedLabelDrawables
 */
export function commit2DLabels(
  updatedLabelDrawables: Array<Readonly<Label2D>>
): void {
  const state = getState()
  const numItems = state.task.items.length
  const updatedShapes: ItemShapeIdMap = {}
  const updatedLabels: ItemLabelIdMap = {}
  const tracking = state.task.config.tracking
  const actions: BaseAction[] = []
  updatedLabelDrawables.forEach((drawable) => {
    drawable.setManual()
    if (drawable.isValid()) {
      // Valid drawable
      if (!drawable.temporary) {
        // Existing drawable
        if (tracking) {
          updateTrack(drawable, updatedLabels, updatedShapes)
        } else {
          updateLabel(drawable, updatedLabels, updatedShapes)
        }
      } else {
        // New drawable
        if (tracking) {
          // Add track
          actions.push(addNewTrack(drawable, numItems))
        } else {
          // Add labels
          actions.push(addNewLabel(drawable))
        }
      }
    } else {
      // Invalid drawable
      if (!drawable.temporary) {
        // Existing drawable
        if (tracking) {
          actions.push(terminateTrackFromDrawable(drawable, numItems))
        } else {
          actions.push(deleteInvalidLabel(drawable))
        }
      }
      // New invalid drawable should be dropped. nothing happens.
    }
  })
  actions.push(commitLabelsToState(updatedLabels))
  actions.push(commitShapesToState(updatedShapes))
  dispatch(makeSequential(actions, true))
}

/**
 * Commit labels to state
 *
 * @param updatedLabelDrawables
 * @param tracking
 */
export function commitLabels(
  updatedLabelDrawables: Array<Readonly<Label2D | Label3D>>,
  tracking: boolean
): void {
  // Get labels & tracks to commit indexed by itemIndex
  const updatedShapes: ItemShapeIdMap = {}
  const updatedLabels: ItemLabelIdMap = {}
  const newTracks: Track[] = []
  const newLabels: Array<Readonly<Label2D | Label3D>> = []
  const state = getState()
  const numItems = state.task.items.length
  const actions: BaseAction[] = []
  updatedLabelDrawables.forEach((drawable) => {
    drawable.setManual()
    if (!drawable.temporary) {
      // Existing labels & tracks
      if (tracking) {
        if (drawable.trackId in Session.tracks) {
          const track = Session.tracks[drawable.trackId]
          track.update(drawable.label.item, drawable)
          for (const index of track.updatedIndices) {
            if (!(index in updatedShapes)) {
              updatedShapes[index] = {}
            }
            const shapes = track.getShapes(index)
            for (const shape of shapes) {
              updatedShapes[index][shape.id] = shape
            }

            if (!(index in updatedLabels)) {
              updatedLabels[index] = {}
            }
            const label = track.getLabel(index)
            if (label !== null) {
              updatedLabels[index][label.id] = label
            }
          }
          track.clearUpdatedIndices()
        }
      } else {
        const shapes = drawable.shapes()
        if (!(drawable.item in updatedShapes)) {
          updatedShapes[drawable.item] = {}
        }
        for (const shape of shapes) {
          updatedShapes[drawable.item][shape.id] = shape
        }

        if (!(drawable.item in updatedLabels)) {
          updatedLabels[drawable.item] = {}
        }
        updatedLabels[drawable.item][drawable.labelId] = drawable.label
      }
    } else {
      // New labels and tracks
      if (tracking) {
        const track = new Track()
        let parentTrack
        if (
          drawable.parent !== null &&
          drawable.parent.trackId in Session.tracks
        ) {
          parentTrack = Session.tracks[drawable.parent.trackId]
        }
        track.init(
          drawable.item,
          drawable,
          numItems - drawable.item + 1,
          parentTrack
        )
        newTracks.push(track)
      } else {
        newLabels.push(drawable)
      }
    }
  })

  if (tracking && newTracks.length > 0) {
    // Add new tracks to state
    for (const track of newTracks) {
      const indices = []
      const labels = []
      const types = []
      const shapes = []
      for (let i = 0; i < numItems; i++) {
        const label = track.getLabel(i)
        const currentTypes = []
        const currentShapes = []
        if (label !== null) {
          const indexedShapes = track.getShapes(i)
          for (const indexedShape of indexedShapes) {
            currentTypes.push(indexedShape.shapeType)
            currentShapes.push(indexedShape)
          }
          indices.push(i)
          labels.push(label)
          types.push(currentTypes)
          shapes.push(currentShapes)
        }
      }
      actions.push(addTrack(indices, track.type, labels, shapes))
    }
  } else if (!tracking && newLabels.length > 0) {
    // Add new labels to state
    const labels = []
    const shapes = []
    for (const label of newLabels) {
      labels.push(label.label)
      const shapeStates = label.shapes()
      shapes.push(shapeStates)
    }
    actions.push(addLabelsToItem(newLabels[0].item, labels, shapes))
  }

  actions.push(commitShapesToState(updatedShapes))
  actions.push(commitLabelsToState(updatedLabels))
  Session.label3dList.clearUpdatedLabels()
  dispatch(makeSequential(actions))
}
