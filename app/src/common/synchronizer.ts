import OrderedMap from "orderedmap"

import {
  makeSequential,
  setStatusAfterConnect,
  setStatusToComputeDone,
  setStatusToComputing,
  setStatusToReconnecting,
  setStatusToSaved,
  setStatusToSaving,
  setStatusToSubmitted,
  setStatusToSubmitting,
  setStatusToUnsaved,
  updateTask
} from "../action/common"
import * as actionConsts from "../const/action"
import { EventName } from "../const/connection"
import { isSessionFullySaved } from "../functional/selector"
import { SocketClient } from "../server/socket_interface"
import * as actionTypes from "../types/action"
import {
  ActionPacketType,
  RegisterMessageType,
  SyncActionMessageType
} from "../types/message"
import { ThunkDispatchType } from "../types/redux"
import { State } from "../types/state"
import Session from "./session"
import { setupSession } from "./session_setup"
import { uid } from "./uid"
import { doesPacketTriggerModel, index2str } from "./util"

const CONFIRMATION_MESSAGE =
  "You have unsaved changes that will be lost if you leave this page. "

/**
 * Synchronizes data with other sessions
 */
export class Synchronizer {
  /**
   * Getter for number of logged (acked) actions
   */
  public get numLoggedActions(): number {
    return this.actionLog.length
  }

  /**
   * Get number of actions in the process of being saved
   */
  public get numActionsPendingSave(): number {
    return this.actionsPendingSave.size
  }

  /**
   * Getter for number of actions with predictions running
   */
  public get numActionsPendingPrediction(): number {
    return this.actionsPendingPrediction.size
  }

  /** Socket connection */
  public socket: SocketClient
  /** Name of the project */
  public projectName: string
  /** Index of the task */
  public taskIndex: number
  /** The user/browser id, constant across sessions */
  public userId: string
  /** Actions queued to be sent to the backend */
  public actionQueue: actionTypes.BaseAction[]
  /**
   * Actions in the process of being saved, mapped by packet id
   * OrderedMap ensures that resending keeps the same order
   */
  private actionsPendingSave: OrderedMap<ActionPacketType>
  /** Timestamped log for completed actions */
  private readonly actionLog: actionTypes.BaseAction[]
  /** Log of packets that have been acked */
  private readonly ackedPackets: Set<string>
  /** The ids of action packets pending model predictions */
  private readonly actionsPendingPrediction: Set<string>
  /** Flag for initial registration completion */
  private registeredOnce: boolean
  /** Name of the DOM container */
  private readonly containerName: string

  /**
   * Constructor
   *
   * @param socket
   * @param taskIndex
   * @param projectName
   * @param userId
   * @param containerName
   */
  constructor(
    socket: SocketClient,
    taskIndex: number,
    projectName: string,
    userId: string,
    containerName: string = ""
  ) {
    this.socket = socket
    this.taskIndex = taskIndex
    this.projectName = projectName
    this.containerName = containerName

    this.actionQueue = []
    this.actionsPendingSave = OrderedMap.from()
    this.actionLog = []
    this.userId = userId
    this.ackedPackets = new Set()
    this.actionsPendingPrediction = new Set()
    this.registeredOnce = false

    window.onbeforeunload = this.warningPopup.bind(this)
  }

  /**
   * Queue a new action for saving
   *
   * @param action
   * @param autosave
   * @param sessionId
   * @param bots
   * @param dispatch
   */
  public queueActionForSaving(
    action: actionTypes.BaseAction,
    autosave: boolean,
    sessionId: string,
    bots: boolean,
    dispatch: ThunkDispatchType
  ): void {
    const shouldBeSaved = (a: actionTypes.BaseAction): boolean => {
      return (
        sessionId === a.sessionId &&
        ((a.frontendOnly !== undefined && !a.frontendOnly) ||
          a.frontendOnly === undefined) &&
        !actionConsts.isSessionAction(a)
      )
    }
    const actions: actionTypes.BaseAction[] = []
    if (action.type === actionConsts.SEQUENTIAL) {
      actions.push(
        ...(action as actionTypes.SequentialAction).actions.filter(
          (a: actionTypes.BaseAction) => shouldBeSaved(a)
        )
      )
    } else {
      if (shouldBeSaved(action)) {
        actions.push(action)
      }
    }
    if (actions.length > 0) {
      this.actionQueue.push(...actions)
      if (autosave) {
        this.save(sessionId, bots, dispatch)
      } else {
        dispatch(setStatusToUnsaved())
      }
    }
  }

  /**
   * Displays pop-up warning user when leaving with unsaved changes
   *
   * @param e
   */
  public warningPopup(e: BeforeUnloadEvent): string | null {
    const state = Session.getState()
    const reduxState = Session.store.getState()
    if (!state.task.config.autosave && !isSessionFullySaved(reduxState)) {
      e.returnValue = CONFIRMATION_MESSAGE // Gecko + IE
      return CONFIRMATION_MESSAGE // Gecko + Webkit, Safari, Chrome etc.
    } else {
      return null
    }
  }

  /**
   * Registers the session with the backend, triggering a register ack
   *
   * @param sessionId
   * @param dispatch
   */
  public sendConnectionMessage(
    sessionId: string,
    dispatch: ThunkDispatchType
  ): void {
    const message: RegisterMessageType = {
      projectName: this.projectName,
      taskIndex: this.taskIndex,
      sessionId,
      userId: this.userId,
      address: location.origin,
      bot: false
    }
    /* Send the registration message to the backend */
    this.socket.emit(EventName.REGISTER, message)
    dispatch(setStatusAfterConnect())
  }

  /**
   * Initialized synced state, and sends any queued actions
   *
   * @param state
   * @param autosave
   * @param sessionId
   * @param bots
   * @param dispatch
   */
  public finishRegistration(
    state: State,
    autosave: boolean,
    sessionId: string,
    bots: boolean,
    dispatch: ThunkDispatchType
  ): void {
    if (!this.registeredOnce) {
      // One-time setup after first registration
      this.registeredOnce = true
      setupSession(state, this.containerName)
    } else {
      // Get the local session in-sync after a disconnect/reconnect
      if (autosave) {
        const actions: actionTypes.BaseAction[] = []
        // Update with any backend changes that occurred during disconnect
        actions.push(updateTask(state.task))

        // Re-apply frontend task actions after updating task from backend
        for (const actionPacket of this.listActionsPendingSave()) {
          for (const action of actionPacket.actions) {
            if (actionConsts.isTaskAction(action)) {
              action.frontendOnly = true
              actions.push(action)
            }
          }
        }
        dispatch(makeSequential(actions))
      }

      for (const actionPacket of this.listActionsPendingSave()) {
        this.sendActions(actionPacket, sessionId, dispatch)
      }
      if (autosave) {
        this.save(sessionId, bots, dispatch)
      }
    }
  }

  /**
   * Called when backend sends ack for actions that were sent to be synced
   * Updates relevant queues and syncs actions from other sessions
   *
   * @param message
   * @param sessionId
   * @param dispatch
   */
  public handleBroadcast(
    message: SyncActionMessageType,
    sessionId: string,
    dispatch: ThunkDispatchType
  ): void {
    const actionPacket = message.actions
    // Remove stored actions when they are acked
    this.actionsPendingSave = this.actionsPendingSave.remove(actionPacket.id)

    const actions: actionTypes.BaseAction[] = []

    // If action was already acked, ignore it
    if (this.ackedPackets.has(actionPacket.id)) {
      return
    }
    this.ackedPackets.add(actionPacket.id)

    for (const action of actionPacket.actions) {
      // ActionLog matches backend action ordering
      this.actionLog.push(action)
      if (action.sessionId !== sessionId) {
        if (actionConsts.isTaskAction(action)) {
          // Dispatch any task actions broadcasted from other sessions
          actions.push(action)
        }
      }
    }

    if (this.actionsPendingPrediction.has(actionPacket.id)) {
      /* Original action was acked by the server
       * This means the bot also received the action
       * And started its prediction */
      actions.push(setStatusToComputing())
    } else if (
      actionPacket.triggerId !== undefined &&
      this.actionsPendingPrediction.has(actionPacket.triggerId)
    ) {
      // Ack of bot action means prediction is finished
      this.actionsPendingPrediction.delete(actionPacket.triggerId)
      if (this.actionsPendingPrediction.size === 0) {
        dispatch(setStatusToComputeDone())
      }
    } else if (message.sessionId === sessionId) {
      if (actionConsts.hasSubmitAction(actionPacket.actions)) {
        dispatch(setStatusToSubmitted())
      } else if (this.actionsPendingSave.size === 0) {
        // Once all actions being saved are acked, update the save status
        dispatch(setStatusToSaved())
      }
    }
    if (actions.length > 0) {
      dispatch(makeSequential(actions))
    }
  }

  /**
   * Called when session disconnects from backend
   *
   * @param dispatch
   */
  public handleDisconnect(dispatch: ThunkDispatchType): void {
    dispatch(setStatusToReconnecting())
  }

  /**
   * Converts ordered map of action packets to a list
   * Order of the list should match the order in which keys were added
   */
  public listActionsPendingSave(): ActionPacketType[] {
    const values: ActionPacketType[] = []
    if (this.actionsPendingSave.size > 0) {
      this.actionsPendingSave.forEach(
        (_key: string, value: ActionPacketType) => {
          values.push(value)
        }
      )
    }
    return values
  }

  /**
   * Send all queued actions to the backend
   * and move actions to actionsPendingSave
   *
   * @param sessionId
   * @param bots
   * @param dispatch
   */
  public save(
    sessionId: string,
    bots: boolean,
    dispatch: ThunkDispatchType
  ): void {
    if (this.socket.connected) {
      if (this.actionQueue.length > 0) {
        const packet: ActionPacketType = {
          actions: this.actionQueue,
          id: uid()
        }
        this.actionsPendingSave = this.actionsPendingSave.update(
          packet.id,
          packet
        )
        if (doesPacketTriggerModel(packet, bots)) {
          this.actionsPendingPrediction.add(packet.id)
        }
        this.sendActions(packet, sessionId, dispatch)
        this.actionQueue = []
      }
    }
  }

  /**
   * Send given action packet to the backend
   * Can be called multiple times if previous attempts aren't acked
   *
   * @param actionPacket
   * @param sessionId
   * @param dispatch
   */
  public sendActions(
    actionPacket: ActionPacketType,
    sessionId: string,
    dispatch: ThunkDispatchType
  ): void {
    const message: SyncActionMessageType = {
      taskId: index2str(this.taskIndex),
      projectName: this.projectName,
      sessionId,
      actions: actionPacket,
      bot: false
    }
    this.socket.emit(EventName.ACTION_SEND, message)
    if (actionConsts.hasSubmitAction(actionPacket.actions)) {
      dispatch(setStatusToSubmitting())
    } else {
      dispatch(setStatusToSaving())
    }
  }
}

export default Synchronizer
