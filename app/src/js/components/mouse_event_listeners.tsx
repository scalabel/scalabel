import * as React from 'react'
import EventListener, { withOptions } from 'react-event-listener'
import * as LabelTypes from '../common/label_types'
import Session from '../common/session'

type mouseEventListener = (e: MouseEvent) => void

interface MouseEventListenersProps {
  /** mouse down event listener */
  onMouseDown: mouseEventListener
  /** mouse move event listener */
  onMouseMove: mouseEventListener
  /** mouse up event listener */
  onMouseUp: mouseEventListener
  /** mouse leave event listener */
  onMouseLeave: mouseEventListener
  /** double click event listener */
  onDblClick: mouseEventListener
  /** mouse wheel event listener */
  onWheel: (e: WheelEvent) => void
}

/**
 * wrapper stateless component for mouse event listeners, assigning listeners
 * based on label type
 * @param props
 */
export default function mouseEventListeners (props: MouseEventListenersProps) {
  let eventListeners = null
  const state = Session.getState()
  const labelType = state.task.config.labelTypes[state.user.select.labelType]
  if (labelType !== LabelTypes.TAG) {
    eventListeners = (
      <EventListener
        target='parent'
        onMouseDown={(e) => props.onMouseDown(e)}
        onMouseMove={(e) => props.onMouseMove(e)}
        onMouseUp={(e) => props.onMouseUp(e)}
        onMouseLeave={(e) => props.onMouseLeave(e)}
        onDblClick={(e) => props.onDblClick(e)}
        onWheel={withOptions((e) => props.onWheel(e), { passive: false })}
      />
    )
  }
  return eventListeners
}
