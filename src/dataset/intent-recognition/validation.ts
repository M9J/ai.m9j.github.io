const validation = [
  // "Clear" Intent
  { label: [1, 0, 0, 0, 0, 0, 0], text: "delete everything" },
  { label: [1, 0, 0, 0, 0, 0, 0], text: "start fresh" },
  { label: [1, 0, 0, 0, 0, 0, 0], text: "remove all" },

  // "Hi" Intent
  { label: [0, 1, 0, 0, 0, 0, 0], text: "what's up" },
  { label: [0, 1, 0, 0, 0, 0, 0], text: "good day" },
  { label: [0, 1, 0, 0, 0, 0, 0], text: "salutations" },

  // "Preview" Intent
  { label: [0, 0, 1, 0, 0, 0, 0], text: "show me" },
  { label: [0, 0, 1, 0, 0, 0, 0], text: "can I see" },
  { label: [0, 0, 1, 0, 0, 0, 0], text: "give me a look" },

  // "Theme" Intent
  { label: [0, 0, 0, 1, 0, 0, 0], text: "customize design" },
  { label: [0, 0, 0, 1, 0, 0, 0], text: "change layout" },
  { label: [0, 0, 0, 1, 0, 0, 0], text: "select a palette" },

  // "Reload" Intent
  { label: [0, 0, 0, 0, 1, 0, 0], text: "restart now" },
  { label: [0, 0, 0, 0, 1, 0, 0], text: "try again" },
  { label: [0, 0, 0, 0, 1, 0, 0], text: "begin anew" },

  // "Version" Intent
  { label: [0, 0, 0, 0, 0, 1, 0], text: "which build" },
  { label: [0, 0, 0, 0, 0, 1, 0], text: "current release" },
  { label: [0, 0, 0, 0, 0, 1, 0], text: "software info" },

  // "Help" Intent
  { label: [0, 0, 0, 0, 0, 0, 1], text: "need assistance" },
  { label: [0, 0, 0, 0, 0, 0, 1], text: "where's the manual" },
  { label: [0, 0, 0, 0, 0, 0, 1], text: "guide me" },
];

export default validation;
