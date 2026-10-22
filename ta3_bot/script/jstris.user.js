// ==UserScript==
// @name         Tetris AI 3 Jstris bridge
// @namespace    ta3
// @version      0.2.7
// @description  Sends the local Jstris game state to Tetris AI 3.
// @match        https://jstris.jezevec10.com/*
// @match        https://*.jstris.jezevec10.com/*
// @grant        none
// @run-at       document-start
// ==/UserScript==

(() => {
    "use strict";

    const PORT = 18473;
    const VERSION = 6;
    const WIDTH = 10;
    const VISIBLE_ROWS = 20;
    const HIDDEN_ROWS = 3;
    const MESSAGE_SIZE = 12 + WIDTH * 4;
    const BOARD_REQUEST_SIZE = 4;
    const MOVE_HEADER_SIZE = 24;
    const EMPTY_PIECE = 0xff;
    const RECONNECT_DELAY_MS = 1000;
    const MOVE_NAMES = ["left", "right", "soft drop", "rotate left", "rotate right", "hard drop", "hold"];
    const CONTROL_INDICES = [0, 1, 2, 4, 5, 3, 6];

    let socket = null;
    let game = null;
    let unreadableStateReported = false;
    let moveQueue = Promise.resolve();
    let gameOver = false;
    let pendingBoardRequest = false;
    let lastDropAt = -Infinity;

    console.info("TA3 bridge script loaded");

    /** Jstris pauses when the page loses focus or visibility, so report it as always focused and visible */
    function keepPageActive() {
        Object.defineProperty(Document.prototype, "hidden", {get: () => false, configurable: true});
        Object.defineProperty(Document.prototype, "webkitHidden", {get: () => false, configurable: true});
        Object.defineProperty(Document.prototype, "visibilityState", {get: () => "visible", configurable: true});
        Document.prototype.hasFocus = () => true;

        // capturing on window at document-start runs before every Jstris listener; element blurs (inputs, chat) pass
        const swallow = event => {
            if(event.target === window || event.target === document)
                event.stopImmediatePropagation();
        };
        for(const type of ["blur", "visibilitychange", "webkitvisibilitychange"])
            window.addEventListener(type, swallow, true);
    }

    function toArray(value) {
        if(value && typeof value.toArray === "function")
            return value.toArray();
        if(value && Array.isArray(value._data))
            return value._data;
        return value;
    }

    function pieceId(value, empty = false) {
        value = toArray(value);
        if(value === null || value === undefined)
            return empty ? EMPTY_PIECE : null;
        if(empty && value === 0)
            return EMPTY_PIECE;

        if(Array.isArray(value)) {
            if(value.length < 2 || Number(value[0]) !== 0)
                return null;
            value = value[1];
        }
        else if(typeof value === "object") {
            const set = value.setID ?? value.set ?? value.blockSet ?? 0;
            if(Number(set) !== 0)
                return null;
            value = value.id ?? value.pieceID ?? value.pieceId;
        }

        const id = Number(value);
        return Number.isInteger(id) && 0 <= id && id < 7 ? id : null;
    }

    function getBoard(current) {
        const board = toArray(current.matrix);
        if(!Array.isArray(board) || board.length !== VISIBLE_ROWS)
            return null;
        if(board.some(row => !Array.isArray(toArray(row)) || toArray(row).length !== WIDTH))
            return null;
        return board.map(toArray);
    }

    function getCurrent(current) {
        return pieceId(
            typeof current.getBlock === "function"
                ? current.getBlock()
                : current.block ?? current.activeBlock ?? current.active
        );
    }

    function getHeld(current) {
        return pieceId(
            typeof current.getHold === "function"
                ? current.getHold()
                : current.blockInHold ?? current.hold ?? current.heldBlock,
            true
        );
    }

    function getQueue(current) {
        const queue = toArray(
            typeof current.queueGet === "function" ? current.queueGet() : current.queue
        );
        if(!Array.isArray(queue) || queue.length < 5)
            return null;

        const ids = queue.slice(0, 5).map(piece => pieceId(piece));
        return ids.every(id => id !== null) ? ids : null;
    }

    function encodeState(current, gameOver) {
        const board = getBoard(current);
        const active = getCurrent(current);
        const held = getHeld(current);
        const queue = getQueue(current);
        if(!board || active === null || held === null || !queue)
            return null;

        const data = new ArrayBuffer(MESSAGE_SIZE);
        const bytes = new Uint8Array(data);
        const view = new DataView(data);
        bytes.set(["T".charCodeAt(0), "A".charCodeAt(0), "3".charCodeAt(0), VERSION]);
        bytes[4] = gameOver ? 1 : 0;
        bytes[5] = active;
        bytes[6] = held;
        bytes.set(queue, 7);

        for(let x = 0; x < WIDTH; ++x) {
            let column = 0;
            for(let y = 0; y < VISIBLE_ROWS; ++y) {
                if(Number(board[y][x]) !== 0)
                    column = (column | 1 << (y + HIDDEN_ROWS)) >>> 0;
            }
            view.setUint32(12 + x * 4, column, true);
        }
        return data;
    }

    /** @returns true once the state was sent */
    function sendState(current = game, gameOver = false) {
        game = current;
        if(!game || !socket || socket.readyState !== WebSocket.OPEN)
            return false;

        try {
            const state = encodeState(game, gameOver);
            if(!state) {
                if(!unreadableStateReported) {
                    console.warn("TA3 bridge could not read the Jstris game state");
                    unreadableStateReported = true;
                }
                return false;
            }

            unreadableStateReported = false;
            socket.send(state);
            return true;
        }
        catch(error) {
            console.warn("TA3 bridge failed to read Jstris state", error);
            return false;
        }
    }

    /** the bot waits for exactly one reply per request, so an unreadable state defers the reply instead of dropping it */
    function answerBoardRequest() {
        pendingBoardRequest = !sendState(game, gameOver);
        if(pendingBoardRequest)
            console.debug("TA3 bridge deferred board request until the game state is readable");
    }

    function flushBoardRequest() {
        if(pendingBoardRequest)
            answerBoardRequest();
    }

    async function executeMove(current, move, pressDuration, pressDeviation) {
        const controlIndex = CONTROL_INDICES[move];
        if(controlIndex === undefined)
            throw new Error(`invalid TA3 move ${move}`);

        const binding = current.Settings?.controls?.[controlIndex];
        if(binding === undefined)
            throw new Error(`missing Jstris control binding for ${MOVE_NAMES[move]}`);

        if(move === 5) {
            lastDropAt = performance.now();
            console.info(
                `TA3 bridge placing piece ${current.activeBlock?.id} at ` +
                `x=${current.activeBlock?.pos?.x}, y=${current.activeBlock?.pos?.y}, ` +
                `rotation=${current.activeBlock?.rot}`
            );
        }

        const dispatch = type => {
            const event = new KeyboardEvent(type, {
                bubbles: true,
                cancelable: true,
                code: typeof binding === "string" ? binding : ""
            });
            if(typeof binding === "number") {
                Object.defineProperty(event, "keyCode", {get: () => binding});
                Object.defineProperty(event, "which", {get: () => binding});
            }
            document.dispatchEvent(event);
        };

        dispatch("keydown");
        await wait(randomDelay(pressDuration, pressDeviation));
        dispatch("keyup");
    }

    function wait(delay) {
        return new Promise(resolve => setTimeout(resolve, delay));
    }

    function randomDelay(mean, deviation) {
        let a = 0;
        let b = 0;
        while(a === 0)
            a = Math.random();
        while(b === 0)
            b = Math.random();

        const standardNormal = Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
        return Math.max(0, Math.round(mean + standardNormal * deviation));
    }

    async function executeMoves(data) {
        const bytes = new Uint8Array(data);
        if(bytes.length < BOARD_REQUEST_SIZE || bytes[0] !== "T".charCodeAt(0) ||
           bytes[1] !== "A".charCodeAt(0) || bytes[2] !== "3".charCodeAt(0) || bytes[3] !== VERSION) {
            const header = Array.from(bytes.subarray(0, BOARD_REQUEST_SIZE));
            throw new Error(
                `invalid TA3 move message: ${bytes.length} bytes, header [${header.join(", ")}], ` +
                `expected [84, 65, 51, ${VERSION}]`
            );
        }
        if(bytes.length === BOARD_REQUEST_SIZE) {
            console.debug("TA3 bridge received board request");
            answerBoardRequest();
            return;
        }
        if(!game)
            return;
        if(bytes.length < MOVE_HEADER_SIZE)
            throw new Error("invalid TA3 move message");

        const view = new DataView(data);
        const initialDelay = view.getUint32(4, true);
        const delay = view.getUint32(8, true);
        const deviation = view.getUint32(12, true);
        const pressDuration = view.getUint32(16, true);
        const pressDeviation = view.getUint32(20, true);
        const moveCount = bytes.length - MOVE_HEADER_SIZE;
        console.info(
            `TA3 bridge received ${moveCount} move${moveCount === 1 ? "" : "s"} ` +
            `(initial ${initialDelay}, move ${delay} ± ${deviation} ms, press ${pressDuration} ± ${pressDeviation} ms)`
        );

        // the initial delay counts from the last hard drop, so time spent on the round trip and search is included
        await wait(Math.max(0, lastDropAt + randomDelay(initialDelay, deviation) - performance.now()));
        for(let i = MOVE_HEADER_SIZE; i < bytes.length; ++i) {
            console.debug(`TA3 bridge executing ${MOVE_NAMES[bytes[i]] ?? `unknown move ${bytes[i]}`}`);
            await executeMove(game, bytes[i], pressDuration, pressDeviation);
            if(i + 1 < bytes.length)
                await wait(randomDelay(delay, deviation));
        }
        console.info("TA3 bridge executed move batch");
    }

    function connect() {
        if(socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING))
            return;

        socket = new WebSocket(`ws://127.0.0.1:${PORT}`);
        socket.binaryType = "arraybuffer";
        socket.addEventListener("open", () => {
            console.info("TA3 bridge connected");
        });
        socket.addEventListener("message", event => {
            console.debug("TA3 bridge received a WebSocket message");
            moveQueue = moveQueue
                .then(async () => {
                    if(typeof event.data === "string")
                        return;

                    const data = typeof event.data.arrayBuffer === "function"
                        ? await event.data.arrayBuffer()
                        : event.data;
                    await executeMoves(data);
                })
                .catch(error => console.warn("TA3 bridge failed to execute moves", error));
        });
        socket.addEventListener("close", event => {
            console.warn(`TA3 bridge disconnected (${event.code}${event.reason ? `: ${event.reason}` : ""})`);
            socket = null;
            // a reply meant for the old connection would desync the next request/reply pair
            pendingBoardRequest = false;
            setTimeout(connect, RECONNECT_DELAY_MS);
        });
        socket.addEventListener("error", () => {});
    }

    function wrap(prototype, name, callback, before = false) {
        const original = prototype[name];
        if(typeof original !== "function")
            return;

        prototype[name] = function(...args) {
            if(before)
                callback(this);
            const result = original.apply(this, args);
            if(!before)
                callback(this);
            return result;
        };
    }

    function install() {
        if(typeof Game !== "function" || Game.prototype.__ta3BridgeInstalled)
            return false;

        Object.defineProperty(Game.prototype, "__ta3BridgeInstalled", {value: true});
        wrap(Game.prototype, "updateQueueBox", current => {
            game = current;
            queueMicrotask(flushBoardRequest);
        });
        wrap(Game.prototype, "readyGo", current => queueMicrotask(() => {
            game = current;
            gameOver = false;
            flushBoardRequest();
        }));
        wrap(Game.prototype, "GameOver", current => {
            game = current;
            gameOver = true;
            flushBoardRequest();
        }, true);
        console.info("TA3 bridge installed Jstris hooks");
        return true;
    }

    keepPageActive();
    connect();
    if(!install()) {
        const installer = setInterval(() => {
            if(install())
                clearInterval(installer);
        }, 50);
    }
})();
