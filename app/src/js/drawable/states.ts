import { addTrack, changeLabelsProps } from '../action/common'
import { ADD_LABELS, CHANGE_SHAPES } from '../action/types'
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
        const [ids,,shapes] = drawable.shapeStates()
        if (!(drawable.item in updatedShapes)) {
          updatedShapes[drawable.item] = {}
        }
        for (let i = 0; i < ids.length; i++) {
          updatedShapes[drawable.item][ids[i]] = shapes[i]
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
            Session.numItems - drawable.item + 1,
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
      for (let i = 0; i < Session.numItems; i++) {
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
    const types = []
    const shapes = []
    for (const label of newLabels) {
      labels.push(label.label)
      const [, shapeTypes, shapeStates] = label.shapeStates()
      types.push(shapeTypes)
      shapes.push(shapeStates)
    }
    Session.dispatch(
      {
        type: ADD_LABELS,
        sessionId: Session.id,
        itemIndices: [newLabels[0].item],
        labels: [labels],
        shapes: [shapes]
      }
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
    Session.dispatch(
      {
        type: CHANGE_SHAPES,
        sessionId: Session.id,
        itemIndices,
        shapeIds,
        shapes
      }
    )
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
