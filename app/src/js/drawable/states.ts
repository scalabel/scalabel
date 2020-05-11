import { addLabelsToItem, addTrack, changeLabelsProps, changeShapesInItems } from '../action/common'
import Session from '../common/session'
import { Track } from '../common/track/track'
import { LabelIdMap, ShapeIdMap } from '../functional/types'
import Label2D from './2d/label2d'
import Label3D from './3d/label3d'

/**
 * Commit labels to state
 */
export function commitLabels (
  updatedLabelDrawables: Array<Readonly<Label2D | Label3D>>
) {
  // Get labels & tracks to commit indexed by itemIndex
  const updatedShapes: { [index: number]: ShapeIdMap } = {}
  const updatedLabels: { [index: number]: LabelIdMap} = {}
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

  if (Object.keys(updatedShapes).length > 0) {
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

  if (Object.keys(updatedLabels).length > 0) {
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
  Session.label3dList.clearUpdatedLabels()
}
