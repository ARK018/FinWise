// FinWise Banking Assistant JavaScript

document.addEventListener("DOMContentLoaded", function () {
  // DOM elements
  const chatMessages = document.getElementById("chat-messages");
  const userInput = document.getElementById("user-input");
  const sendButton = document.getElementById("send-button");
  const toggleDetailsButton = document.getElementById("toggle-details");
  const queryDetails = document.getElementById("query-details");
  const detectedCategory = document.getElementById("detected-category");
  const confidenceElement = document.getElementById("confidence");
  const relevantSources = document.getElementById("relevant-sources");
  const sampleQuestions = document.querySelectorAll(".sample-question");

  // Track if the assistant is currently processing a request
  let isProcessing = false;

  // Handle sending messages
  function sendMessage() {
    const message = userInput.value.trim();

    if (message === "" || isProcessing) return;

    // Add user message to chat
    addUserMessage(message);

    // Clear input field
    userInput.value = "";

    // Show typing indicator
    showTypingIndicator();

    // Set processing flag
    isProcessing = true;

    // Send message to backend API
    processUserQuery(message);
  }

  // Add user message to chat
  function addUserMessage(message) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("flex", "justify-end", "message-animation");

    messageDiv.innerHTML = `
            <div class="user-message p-3 max-w-md">
                <p class="text-gray-800">${escapeHtml(message)}</p>
            </div>
        `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
  }

  // Add assistant message to chat
  function addAssistantMessage(message) {
    // Remove typing indicator
    removeTypingIndicator();

    const messageDiv = document.createElement("div");
    messageDiv.classList.add("flex", "items-start", "message-animation");

    messageDiv.innerHTML = `
            <div class="flex-shrink-0 h-10 w-10 rounded-full bg-blue-500 flex items-center justify-center text-white mr-3">
                <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
            </div>
            <div class="bot-message p-3 max-w-3xl">
                <p class="text-gray-800">${escapeHtml(message).replace(
                  /\n/g,
                  "<br>"
                )}</p>
            </div>
        `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    // Reset processing flag
    isProcessing = false;
  }

  // Show typing indicator
  function showTypingIndicator() {
    const indicatorDiv = document.createElement("div");
    indicatorDiv.id = "typing-indicator";
    indicatorDiv.classList.add("flex", "items-start", "message-animation");

    indicatorDiv.innerHTML = `
            <div class="flex-shrink-0 h-10 w-10 rounded-full bg-blue-500 flex items-center justify-center text-white mr-3">
                <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
            </div>
            <div class="bot-message p-3">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

    chatMessages.appendChild(indicatorDiv);
    scrollToBottom();
  }

  // Remove typing indicator
  function removeTypingIndicator() {
    const indicator = document.getElementById("typing-indicator");
    if (indicator) {
      indicator.remove();
    }
  }

  // Process user query through backend API
  async function processUserQuery(query) {
    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();

      if (data.error) {
        addAssistantMessage(
          "I'm sorry, but there was an error processing your request. " +
            data.error
        );
        return;
      }

      // Update assistant response
      addAssistantMessage(data.response);

      // Update query details
      updateQueryDetails(data);
    } catch (error) {
      console.error("Error:", error);
      addAssistantMessage(
        "I'm sorry, but there was an error connecting to the server. Please try again later."
      );
      isProcessing = false;
    }
  }

  // Update query details panel
  function updateQueryDetails(data) {
    // Format the category for display
    const category = data.detected_category.replace(/_/g, " ");
    detectedCategory.innerHTML = `<span class="category-badge category-${data.detected_category}">${category}</span>`;

    // Format confidence as percentage
    const confidencePercent = Math.round(data.confidence * 100);
    confidenceElement.innerHTML = `
            ${confidencePercent}%
            <div class="confidence-meter">
                <div class="confidence-level" style="width: ${confidencePercent}%"></div>
            </div>
        `;

    // Clear previous sources
    relevantSources.innerHTML = "";

    // Add sources with relevance scores
    data.relevant_sources.forEach((source, index) => {
      const sourceDiv = document.createElement("div");
      sourceDiv.classList.add("mb-1", "pb-1", "border-b", "border-gray-100");

      // Format score as percentage
      const scorePercent = Math.round(
        data.source_relevance_scores[index] * 100
      );

      // Truncate source if too long
      const truncatedSource =
        source.length > 120 ? source.substring(0, 120) + "..." : source;

      sourceDiv.innerHTML = `
                <div class="text-gray-700">${escapeHtml(truncatedSource)}</div>
                <div class="flex items-center text-xs text-gray-500 mt-1">
                    <span>Relevance: ${scorePercent}%</span>
                    <div class="ml-2 flex-grow confidence-meter">
                        <div class="confidence-level" style="width: ${scorePercent}%"></div>
                    </div>
                </div>
            `;

      relevantSources.appendChild(sourceDiv);
    });
  }

  // Toggle query details panel
  function toggleDetails() {
    if (queryDetails.classList.contains("hidden")) {
      queryDetails.classList.remove("hidden");
      toggleDetailsButton.textContent = "Hide";
    } else {
      queryDetails.classList.add("hidden");
      toggleDetailsButton.textContent = "Show";
    }
  }

  // Helper function to escape HTML
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Scroll chat to bottom
  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // Event listeners
  sendButton.addEventListener("click", sendMessage);

  userInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      sendMessage();
    }
  });

  toggleDetailsButton.addEventListener("click", toggleDetails);

  // Sample questions functionality
  sampleQuestions.forEach((question) => {
    question.addEventListener("click", function () {
      if (isProcessing) return;

      userInput.value = this.textContent;
      sendMessage();
    });
  });

  // Focus input field on page load
  userInput.focus();
});
