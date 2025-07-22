# AI Safety Daily - {{ metadata.date|format_date('%B %d, %Y') }}

*Your daily digest of AI safety developments*

---

## Today's Highlights

{% for article in articles[:3] %}
**{{ loop.index }}. {{ article.title }}**  
{{ article.summary|truncate_words(30) }}  
[Read more]({{ article.url }}) | {{ article.source_tier|source_tier_emoji }} {{ article.relevance_level|relevance_emoji }}
{% endfor %}

---

## Full Stories

{% for article in articles %}
### {{ loop.index }}. {{ article.title }}

{% if article.summary %}
{{ article.summary }}
{% endif %}

{% if article.key_points %}
**Key takeaways:**
{% for point in article.key_points[:3] %}
• {{ point }}
{% endfor %}
{% endif %}

**Source:** [{{ article.source }}]({{ article.url }}) {{ article.source_tier|source_tier_emoji }}  
**Relevance:** {{ article.relevance_level|relevance_emoji }} {{ article.relevance_level|title }}

---

{% endfor %}

## Tomorrow's Watch

Keep an eye out for developments in:
- AI governance policy announcements
- New safety research publications
- Industry AI alignment initiatives

---

*Generated {{ metadata.generation_time|format_date('%I:%M %p') }} | [Subscribe](mailto:hello@aisafetyturkiye.org) | [Archive](https://aisafetyturkiye.org/archive)*
