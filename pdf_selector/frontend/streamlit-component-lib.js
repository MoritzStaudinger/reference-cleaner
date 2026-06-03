/*
 * Minimal Streamlit component bridge — vanilla JS, no build step.
 *
 * Exposes window.Streamlit with the methods our component needs to talk
 * to the Streamlit Python runtime via postMessage to the parent frame.
 */

(function () {
  const Streamlit = {
    RENDER_EVENT: "streamlit:render",

    _send: function (type, payload) {
      window.parent.postMessage(
        Object.assign({ isStreamlitMessage: true, type: type }, payload || {}),
        "*"
      );
    },

    setComponentReady: function () {
      Streamlit._send("streamlit:componentReady", { apiVersion: 1 });
    },

    setFrameHeight: function (height) {
      if (height == null) height = document.body.scrollHeight;
      Streamlit._send("streamlit:setFrameHeight", { height: height });
    },

    setComponentValue: function (value) {
      Streamlit._send("streamlit:setComponentValue", {
        value: value,
        dataType: "json",
      });
    },

    events: {
      addEventListener: function (eventType, callback) {
        window.addEventListener("message", function (e) {
          if (!e.data || e.data.type !== eventType) return;
          // Render event payload is { type, args, disabled, theme }
          callback({
            detail: {
              args: e.data.args || {},
              disabled: !!e.data.disabled,
              theme: e.data.theme || null,
            },
          });
        });
      },
    },
  };

  window.Streamlit = Streamlit;
})();
