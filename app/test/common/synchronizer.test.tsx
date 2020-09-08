import { cleanup } from "@testing-library/react"

import { setStatusToUnsaved } from "../../src/action/common"
import Session from "../../src/common/session"
import { Synchronizer } from "../../src/common/synchronizer"
import { uid } from "../../src/common/uid"
import { index2str } from "../../src/common/util"
import { EventName } from "../../src/const/connection"
import * as selector from "../../src/functional/selector"
import { updateState } from "../../src/server/util"
import { AddLabelsAction } from "../../src/types/action"
import {
  ActionPacketType,
  RegisterMessageType,
  SyncActionMessageType
} from "../../src/types/message"
import { ThunkDispatchType } from "../../src/types/redux"
import { getInitialState, getRandomBox2dAction } from "../server/util/util"

let sessionId: string
let botSessionId: string
let taskIndex: number
let projectName: string
let userId: string
let autosave: boolean
const socketEmit = jest.fn()
const mockSocket = {
  connected: true,
  emit: socketEmit
}

beforeAll(() => {
  sessionId = "fakeSessId"
  botSessionId = "botSessId"
  taskIndex = 0
  projectName = "testProject"
  userId = "fakeUserId"
  autosave = true
})

const dispatch: ThunkDispatchType = Session.dispatch.bind(
  Session
) as ThunkDispatchType

beforeEach(() => {
  Session.dispatch(setStatusToUnsaved())
})

afterEach(cleanup)
describe("Test synchronizer functionality", () => {
  test("Test correct registration message gets sent", async () => {
    // Since this deals with registration, don't initialize the state
    const initializeState = false
    const sync = startSynchronizer(initializeState)
    sync.sendConnectionMessage("", dispatch)

    // Frontend doesn't have a session id until after registration
    const expectedSessId = ""
    checkConnectMessage(expectedSessId)
    expect(selector.isStatusUnsaved(Session.store.getState())).toBe(true)
  })

  test("Test send-ack loop", async () => {
    const sync = startSynchronizer()
    dispatchAndCheckActions(sync, 1)

    // After ack arrives, no actions are queued anymore
    const ackPackets = sendAcks(sync)
    expect(sync.numActionsPendingSave).toBe(0)
    expect(sync.numLoggedActions).toBe(1)
    expect(selector.isStatusSaved(Session.store.getState())).toBe(true)

    // If second ack arrives, it is ignored
    sync.handleBroadcast(packetToMessage(ackPackets[0]), sessionId, dispatch)
    expect(sync.numLoggedActions).toBe(1)
  })

  test("Test model prediction status", async () => {
    const sync = startSynchronizer()

    dispatchAndCheckActions(sync, 2, true)

    // After acks arrive, session status is marked as computing
    const ackPackets = sendAcks(sync)
    expect(selector.isStatusComputing(Session.store.getState())).toBe(true)

    // Mark computation as finished when all model actions arrive
    const modelPackets = []
    for (const ackPacket of ackPackets) {
      const modelAction = getRandomBox2dAction()
      modelAction.sessionId = botSessionId
      const modelPacket: ActionPacketType = {
        actions: [modelAction],
        id: uid(),
        triggerId: ackPacket.id
      }

      modelPackets.push(modelPacket)
    }

    sync.handleBroadcast(
      packetToMessageBot(modelPackets[0]),
      sessionId,
      dispatch
    )
    expect(sync.numActionsPendingPrediction).toBe(1)
    expect(selector.isComputeDone(Session.store.getState())).toBe(false)

    sync.handleBroadcast(
      packetToMessageBot(modelPackets[1]),
      sessionId,
      dispatch
    )
    expect(sync.numActionsPendingPrediction).toBe(0)
    expect(selector.isComputeDone(Session.store.getState())).toBe(true)
  })

  test("Test reconnection", async () => {
    const sync = startSynchronizer()
    const frontendActions = dispatchAndCheckActions(sync, 1)

    // Backend disconnects instead of acking
    sync.handleDisconnect(dispatch)
    expect(selector.isStatusReconnecting(Session.store.getState())).toBe(true)

    // Reconnect, but some missed actions occured in the backend
    const newInitialState = updateState(getInitialState(sessionId), [
      getRandomBox2dAction()
    ])
    sync.sendConnectionMessage(sessionId, dispatch)
    checkConnectMessage(sessionId)
    sync.finishRegistration(
      newInitialState,
      autosave,
      sessionId,
      false,
      dispatch
    )

    /**
     * Check that frontend state updates correctly
     * Except for session state, which will change because of status effects
     */
    const expectedState = updateState(newInitialState, frontendActions)
    expect(Session.getState().task).toMatchObject(expectedState.task)
    expect(Session.getState().user).toMatchObject(expectedState.user)

    // Also check that save is still in progress
    expect(sync.numActionsPendingSave).toBe(1)
    checkActionsAreSaving(sync, frontendActions)
    expect(selector.isStatusSaving(Session.store.getState())).toBe(true)

    // After ack arrives, no actions are queued anymore
    sendAcks(sync)
    expect(Session.getState().task).toMatchObject(expectedState.task)
    expect(Session.getState().user).toMatchObject(expectedState.user)
    expect(sync.numActionsPendingSave).toBe(0)
    expect(sync.numLoggedActions).toBe(1)
  })
})

/**
 * Dispatch and check the effects of a single add label action
 *
 * @param sync
 * @param numActions
 * @param bots
 */
function dispatchAndCheckActions(
  sync: Synchronizer,
  numActions: number,
  bots: boolean = false
): AddLabelsAction[] {
  // Dispatch actions to trigger sync events
  const actions: AddLabelsAction[] = []
  for (let _ = 0; _ < numActions; _++) {
    const action = getRandomBox2dAction()
    sync.queueActionForSaving(action, autosave, sessionId, bots, dispatch)
    actions.push(action)
  }

  // Verify the synchronizer state before ack arrives
  expect(sync.numActionsPendingSave).toBe(numActions)
  checkActionsAreSaving(sync, actions)
  expect(selector.isStatusSaving(Session.store.getState())).toBe(true)
  if (bots) {
    expect(sync.numActionsPendingPrediction).toBe(numActions)
  }
  return actions
}

/**
 * Acks all the waiting packets
 *
 * @param sync
 */
function sendAcks(sync: Synchronizer): ActionPacketType[] {
  const actionPackets = sync.listActionsPendingSave()
  for (const actionPacket of actionPackets) {
    sync.handleBroadcast(packetToMessage(actionPacket), sessionId, dispatch)
  }
  return actionPackets
}

/**
 * Check that the actions were sent to the backend for saving
 *
 * @param sync
 * @param actions
 */
function checkActionsAreSaving(
  sync: Synchronizer,
  actions: AddLabelsAction[]
): void {
  const actionPackets = sync.listActionsPendingSave()
  expect(actionPackets.length).toBe(actions.length)
  for (let i = 0; i < actions.length; i++) {
    expect(actionPackets[i].actions[0]).toBe(actions[i])
  }
}

/**
 * Checkthat correct connection message was sent
 *
 * @param sessId
 */
function checkConnectMessage(sessId: string): void {
  const expectedMessage: RegisterMessageType = {
    projectName,
    taskIndex,
    sessionId: sessId,
    userId,
    address: location.origin,
    bot: false
  }
  expect(socketEmit).toHaveBeenCalledWith(EventName.REGISTER, expectedMessage)
}

/**
 * Start the browser synchronizer being tested
 *
 * @param setInitialState
 */
function startSynchronizer(setInitialState: boolean = true): Synchronizer {
  const synchronizer = new Synchronizer(
    mockSocket,
    taskIndex,
    projectName,
    userId
  )

  if (setInitialState) {
    const initialState = getInitialState(sessionId)
    synchronizer.finishRegistration(
      initialState,
      initialState.task.config.autosave,
      initialState.session.id,
      initialState.task.config.bots,
      dispatch
    )
  }

  // Initially, no actions are queued for saving
  expect(synchronizer.numActionsPendingSave).toBe(0)
  expect(selector.isStatusUnsaved(Session.store.getState())).toBe(true)

  return synchronizer
}

/**
 * Convert action packet to sync message
 *
 * @param packet
 */
function packetToMessage(packet: ActionPacketType): SyncActionMessageType {
  return {
    actions: packet,
    projectName,
    sessionId,
    taskId: index2str(taskIndex),
    bot: false
  }
}

/**
 * Convert action packet to sync message from a bot
 *
 * @param packet
 */
function packetToMessageBot(packet: ActionPacketType): SyncActionMessageType {
  return {
    actions: packet,
    projectName,
    sessionId: botSessionId,
    taskId: index2str(taskIndex),
    bot: true
  }
}
