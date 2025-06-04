<h2 id="news">News 🗞️</h2>

<div class="news-container">
  <div class="news-list">
    {% for item in site.data.news %}
    <div class="news-item">
      <p><b>[{{ item.date }}]</b> {{ item.content }}</p>
    </div>
    {% endfor %}
  </div>
</div> 