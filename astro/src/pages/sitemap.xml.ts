import { sanity } from "../lib/sanity";

const SITE = "https://www.rutmmag.com";

const STATIC_PAGES = [
  { url: "/", priority: "1.0", changefreq: "daily" },
  { url: "/about", priority: "0.6", changefreq: "monthly" },
  { url: "/newsletter", priority: "0.6", changefreq: "monthly" },
  { url: "/tell-yours", priority: "0.5", changefreq: "monthly" },
];

function toXmlDate(date?: string) {
  if (!date) return new Date().toISOString().split("T")[0];
  return new Date(date).toISOString().split("T")[0];
}

function urlEntry(
  loc: string,
  opts: { lastmod?: string; priority?: string; changefreq?: string } = {}
) {
  return `
  <url>
    <loc>${SITE}${loc}</loc>
    ${opts.lastmod ? `<lastmod>${toXmlDate(opts.lastmod)}</lastmod>` : ""}
    ${opts.changefreq ? `<changefreq>${opts.changefreq}</changefreq>` : ""}
    ${opts.priority ? `<priority>${opts.priority}</priority>` : ""}
  </url>`;
}

export async function GET() {
  const [articles, sections, authors] = await Promise.all([
    sanity.fetch<{ slug: string; publishedAt?: string }[]>(
      `*[_type == "article" && defined(slug.current)]{ "slug": slug.current, publishedAt }`
    ),
    sanity.fetch<{ slug: string }[]>(
      `*[_type == "section" && defined(slug.current)]{ "slug": slug.current }`
    ),
    sanity.fetch<{ slug: string }[]>(
      `*[_type == "author" && defined(slug.current)]{ "slug": slug.current }`
    ),
  ]);

  const staticEntries = STATIC_PAGES.map((p) =>
    urlEntry(p.url, { priority: p.priority, changefreq: p.changefreq })
  );

  const articleEntries = articles.map((a) =>
    urlEntry(`/article/${a.slug}`, {
      lastmod: a.publishedAt,
      priority: "0.8",
      changefreq: "monthly",
    })
  );

  const sectionEntries = sections.map((s) =>
    urlEntry(`/section/${s.slug}`, { priority: "0.7", changefreq: "weekly" })
  );

  const authorEntries = authors.map((a) =>
    urlEntry(`/author/${a.slug}`, { priority: "0.5", changefreq: "monthly" })
  );

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${[...staticEntries, ...articleEntries, ...sectionEntries, ...authorEntries].join("")}
</urlset>`;

  return new Response(xml, {
    headers: {
      "Content-Type": "application/xml",
      "Cache-Control": "public, max-age=3600",
    },
  });
}
