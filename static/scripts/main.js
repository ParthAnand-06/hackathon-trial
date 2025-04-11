function loadQueue(queueName) {
    const contentArea = document.getElementById('contentArea');
    contentArea.innerHTML = `<h3>${queueName} Tickets</h3>`;
  
    // Fetch the tickets for the specific queue from Flask backend
    fetch(`/load-tickets?queue=${queueName}`)
      .then(response => response.json())
      .then(data => {
        if (data.tickets && data.tickets.length) {
          let ticketList = "<ul>";
          data.tickets.forEach(ticket => {
            ticketList += `
              <li>
                <strong>${ticket.subject}</strong><br>
                <em>From: ${ticket.from}</em><br>
                <button onclick="assignTicket('${ticket.ticket_id}')">Assign</button>
              </li>`;
          });
          ticketList += "</ul>";
          contentArea.innerHTML += ticketList;
        } else {
          contentArea.innerHTML += "<p>No tickets available.</p>";
        }
      })
      .catch(error => {
        contentArea.innerHTML += "<p>Error loading tickets.</p>";
        console.error('Error fetching tickets:', error);
      });
  }
  function assignTicket(ticketId) {
    const agent = prompt("Enter agent's name to assign this ticket:");
    if (agent) {
      fetch(`/assign-ticket?ticket_id=${ticketId}&agent=${agent}`, {
        method: "POST",
      })
      .then(response => response.json())
      .then(data => {
        if (data.message) {
          alert(data.message);
          loadQueue(data.queueName); // Refresh the queue after assigning
        }
      })
      .catch(error => {
        console.error('Error assigning ticket:', error);
      });
    }
  }
    