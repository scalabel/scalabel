import { addLabel, addLabelsToItem, addTrack, changeLabelsProps, changeShapesInItems, deleteLabel } from '../action/common'
import { deleteTracks, terminateTracks } from '../action/track'
import Session from '../common/session'
import { Track } from '../common/track/track'
import { LabelIdMap, ShapeIdMap } from '../functional/types'
import Label2D from './2d/label2d'
import Label3D from './3d/label3d'

interface ItemLabelIdMap { [index: number]: LabelIdMap }
interface ItemShapeIdMap { [index: number]: ShapeIdMap }

/**
 * Commit changed shapes to state
 *
 * @param {ItemShapeIdMap} updatedShapes
 */
function commitShapesToState (updatedShapes: ItemShapeIdMap) {
  if (Object.keys(updatedShapes).length === 0) {
    return
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
  Session.dispatch(changeShapesInItems(itemIndices, shapeIds, shapes))
}

/**
 * Commit changed labels to state
 *
 * @param {ItemLabelIdMap} updatedLabels
 */
function commitLabelsToState (updatedLabels: ItemLabelIdMap) {
  if (Object.keys(updatedLabels).length === 0) {
    return
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
  Session.dispatch(changeLabelsProps(itemIndices, labelIds, labels))
}

/**
 * Update track from a label
 *
 * @param {(Readonly<Label2D> | Readonly<Label3D>)} drawable
 * @param {ItemLabelIdMap} updatedLabels
 * @param {ItemShapeIdMap} updatedShapes
 */
function updateTrack (drawable: Readonly<Label2D> | Readonly<Label3D>,
                      updatedLabels: ItemLabelIdMap,
                      updatedShapes: ItemShapeIdMap) {
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
    if (label) {
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
function updateLabel (drawable: Readonly<Label2D> | Readonly<Label3D>,
                      updatedLabels: ItemLabelIdMap,
                      updatedShapes: ItemShapeIdMap) {
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
function addNewTrack (drawable: Readonly<Label2D> | Readonly<Label3D>,
                      numItems: number) {
  const track = new Track()
  let parentTrack
  if (drawable.parent && drawable.parent.trackId in Session.tracks) {
    parentTrack = Session.tracks[drawable.parent.trackId]
  }
  track.init(
    drawable.item,
    drawable,
    numItems - drawable.item + 1,
    parentTrack
  )
  const indices = []
  const labels = []
  const types = []
  const shapes = []
  for (let i = 0; i < numItems; i++) {
    const label = track.getLabel(i)
    if (label) {
      const indexedShapes = track.getShapes(i)
      indices.push(i)
      labels.push(label)
      types.push(indexedShapes.map((s) => s.shapeType))
      shapes.push(indexedShapes.map((s) => s))
    }
  }
  Session.dispatch(addTrack(indices, track.type, labels, shapes))
}

/**
 * Add a new label from a drawable
 *
 * @param {(Readonly<Label2D> | Readonly<Label3D>)} drawable
 */
function addNewLabel (drawable: Readonly<Label2D> | Readonly<Label3D>) {
  Session.dispatch(addLabel(drawable.item, drawable.label, drawable.shapes()))
}

/**
 * Terminate track from current drawable
 *
 * @param {(Readonly<Label2D> | Readonly<Label3D>)} drawable
 * @param {number} numItems
 */
function terminateTrackFromDrawable (
  drawable: Readonly<Label2D> | Readonly<Label3D>,
  numItems: number) {
  const track = Session.getState().task.tracks[drawable.label.track]
  if (drawable.item === 0) {
    Session.dispatch(deleteTracks([track]))
  } else {
    Session.dispatch(terminateTracks([track], drawable.item, numItems))
  }
}

/**
 * Delete an invalid existing label
 *
 * @param {(Readonly<Label2D> | Readonly<Label3D>)} drawable
 */
function deleteInvalidLabel (drawable: Readonly<Label2D> | Readonly<Label3D>) {
  Session.dispatch(deleteLabel(drawable.item, drawable.labelId))
}

/**
 * Commit 2D labels to state
 */
export function commit2DLabels (
  updatedLabelDrawables: Array<Readonly<Label2D>>
) {
  const state = Session.getState()
  const numItems = state.task.items.length
  const updatedShapes: ItemShapeIdMap = {}
  const updatedLabels: ItemLabelIdMap = {}
  updatedLabelDrawables.forEach((drawable) => {
    drawable.setManual()
    if (drawable.isValid()) {
      // valid drawable
      if (!drawable.temporary) {
        // existing drawable
        if (Session.tracking) {
          updateTrack(drawable, updatedLabels, updatedShapes)
        } else {
          updateLabel(drawable, updatedLabels, updatedShapes)
        }
      } else {
        // new drawable
        if (Session.tracking) {
          // add track
          addNewTrack(drawable, numItems)
        } else {
          // add labels
          addNewLabel(drawable)
        }
      }
    } else {
      // invalid drawable
      if (!drawable.temporary) {
        // existing drawable
        if (Session.tracking) {
          terminateTrackFromDrawable(drawable, numItems)
        } else {
          deleteInvalidLabel(drawable)
        }
      }
      // new invalid drawable should be dropped. nothing happens.
    }
  })
  commitLabelsToState(updatedLabels)
  commitShapesToState(updatedShapes)
}

/**
 * Commit labels to state
 */
// TODO: Check 3d approach to revise following method
export function commitLabels (
  updatedLabelDrawables: Array<Readonly<Label2D | Label3D>>
) {
  // Get labels & tracks to commit indexed by itemIndex
  const updatedShapes: ItemShapeIdMap = {}
  const updatedLabels: ItemLabelIdMap = {}
  const newTracks: Track[] = []
  const newLabels: Array<Readonly<Label2D | Label3D>> = []
  const state = Session.getState()
  const numItems = state.task.items.length
  updatedLabelDrawables.forEach((drawable) => {
    drawable.setManual()
    if (!drawable.temporary) {
      // Existing labels & tracks
      if (Session.tracking) {
        if (drawable.trackId in Session.tracks) {
          const track = Session.tracks[drawable.trackId]
          track.update(
            drawable.label.item,
            drawable
          )
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
            if (label) {
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
        updatedLabels[drawable.item][drawable.labelId] =
          drawable.label
      }
    } else {
      // New labels and tracks
      if (Session.tracking) {
        const track = new Track()
        if (track) {
          let parentTrack
          if (
            drawable.parent &&
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
        }
      } else {
        newLabels.push(drawable)
      }
    }
  })

  if (Session.tracking && newTracks.length > 0) {
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
        if (label) {
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
      Session.dispatch(addTrack(
        indices, track.type, labels, shapes
      ))
    }
  } else if (!Session.tracking && newLabels.length > 0) {
    // Add new labels to state
    const labels = []
    const shapes = []
    for (const label of newLabels) {
      labels.push(label.label)
      const shapeStates = label.shapes()
      shapes.push(shapeStates)
    }
    Session.dispatch(addLabelsToItem(newLabels[0].item, labels, shapes)
    )
  }

  commitShapesToState(updatedShapes)
  commitLabelsToState(updatedLabels)
  Session.label3dList.clearUpdatedLabels()
}
